from __future__ import division

import sys
sys.path.append("/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Desktop/nus_interactive_segmentaion/segmentaion_code")

import six
import cv2
import keras
import tensorflow as tf
import numpy as np
import math
from model.fpn import FpnNet
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, TensorBoard
from generator.generator import data_generator_train, data_generator_val, aggregate_files

def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """Collect a confusion matrix.
    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.
    Args:
        pred_labels (iterable of numpy.ndarray): See the table in
            :func:`chainercv.evaluations.eval_semantic_segmentation`.
        gt_labels (iterable of numpy.ndarray): See the table in
            :func:`chainercv.evaluations.eval_semantic_segmentation`.
    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.
    """
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 0
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class**2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion


def calc_semantic_segmentation_iou(confusion):
    """Calculate Intersection over Union with a given confusion matrix.
    The definition of Intersection over Union (IoU) is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.
    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number of pixels
            that are labeled as class :math:`i` by the ground truth and
            class :math:`j` by the prediction.
    Returns:
        numpy.ndarray:
        An array of IoUs for the :math:`n\_class` classes. Its shape is
        :math:`(n\_class,)`.
    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) -
                       np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou

def mean_iou(gt_labels, pred_labels):
    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

    return np.nanmean(iou)


def compile(model):
    sgd = SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-4)
    model.compile(loss = "categorical_crossentropy", optimizer = adam, metrics = [mean_iou])

def train_model(model, input_shape, total_classes, train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list):
    compile(model)

    checkpoint_filepath = "/media/abhinav/Abhinav/seg_weights/fpn_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"
    csv_log_filepath = "/media/abhinav/Abhinav/seg_weights/csv/train.csv"
    tensorboard_filepath = "/media/abhinav/Abhinav/seg_weights/logs/"

    checkpoint = ModelCheckpoint(checkpoint_filepath, verbose=1, save_best_only=True, save_weights_only=True, mode='min')

    csv_logger = CSVLogger(filename=csv_log_filepath,
                           separator=',',
                           append=True)

    terminate_on_nan = TerminateOnNaN()

    tensorboard = TensorBoard(log_dir=tensorboard_filepath, histogram_freq=0,
                              write_graph=True, write_images=True)

    callbacks_list = [checkpoint, csv_logger, terminate_on_nan, tensorboard]

    batch_size = 8
    total_size = 1500

    initial_epoch = 0
    final_epoch = 60

    train_generator = data_generator_train(batch_size, input_shape, total_classes, train_orig_images_list, train_seg_images_list)
    val_generator = data_generator_val(batch_size, input_shape, total_classes, val_orig_images_list, val_seg_images_list)

    model.fit_generator(train_generator,
                        steps_per_epoch = math.ceil(total_size / batch_size),
                        epochs = final_epoch,
                        callbacks = callbacks_list,
                        initial_epoch = initial_epoch,
                        validation_data = val_generator,
                        validation_steps = batch_size
                        )


if __name__ == '__main__':
    train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list = aggregate_files()


    input_shape = (256, 256, 3)
    total_classes = 21

    model = FpnNet(image_size = input_shape, n_classes = total_classes)

    weights_file = "/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Desktop/nus_interactive_segmentaion/segmentaion_code/weights/Resnet50.h5"

    model.load_weights(weights_file, by_name=True, skip_mismatch=True)
    train_model(model, input_shape, total_classes, train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list)



