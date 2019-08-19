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
from keras.preprocessing.image import img_to_array

out_palette = {
		   0 : (  0,   0,   0) ,
           1 : (128,   0,   0) ,
           2 : (  0, 128,   0) ,
           3 : (128, 128,   0) ,
           4 : (  0,   0, 128) ,
           5 : (128,   0, 128) ,
           6 : (  0, 128, 128) ,
           7 : (128, 128, 128) ,
           8 : ( 64,   0,   0) ,
           9 : (192,   0,   0) ,
           10: ( 64, 128,   0) ,
           11: (192, 128,   0) ,
           12: ( 64,   0, 128) ,
           13: (192,   0, 128) ,
           14: ( 64, 128, 128) ,
           15: (192, 128, 128) ,
           16: (  0,  64,   0) ,
           17: (128,  64,   0) ,
           18: (  0, 192,   0) ,
           19: (128, 192,   0) ,
           20: (  0,  64, 128)  }

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


def eval_semantic_segmentation(pred_labels, gt_labels):
    """Evaluate metrics used in Semantic Segmentation.
    This function calculates Intersection over Union (IoU), Pixel Accuracy
    and Class Accuracy for the task of semantic segmentation.
    The definition of metrics calculated by this function is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.
    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{mIoU} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{Pixel Accuracy} =  \
        \\frac \
        {\\sum_{i=1}^k N_{ii}} \
        {\\sum_{i=1}^k \\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Class Accuracy} = \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Mean Class Accuracy} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    The more detailed description of the above metrics can be found in a
    review on semantic segmentation [#]_.
    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.
    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.
    Args:
        pred_labels (iterable of numpy.ndarray): See the table below.
        gt_labels (iterable of numpy.ndarray): See the table below.
    .. csv-table::
        :header: name, shape, dtype, format
        :obj:`pred_labels`, ":math:`[(H, W)]`", :obj:`int32`, \
        ":math:`[0, \#class - 1]`"
        :obj:`gt_labels`, ":math:`[(H, W)]`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"
    Returns:
        dict:
        The keys, value-types and the description of the values are listed
        below.
        * **iou** (*numpy.ndarray*): An array of IoUs for the \
            :math:`n\_class` classes. Its shape is :math:`(n\_class,)`.
        * **miou** (*float*): The average of IoUs over classes.
        * **pixel_accuracy** (*float*): The computed pixel accuracy.
        * **class_accuracy** (*numpy.ndarray*): An array of class accuracies \
            for the :math:`n\_class` classes. \
            Its shape is :math:`(n\_class,)`.
        * **mean_class_accuracy** (*float*): The average of class accuracies.
    """
    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

    return {'iou': iou, 'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy)}

if __name__ == '__main__':
    train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list = aggregate_files()


    input_shape = (256, 256, 3)
    total_classes = 21
    subtract_mean = [123, 117, 104]

    model = FpnNet(image_size = input_shape, n_classes = total_classes)

    weights_file = "/media/abhinav/Abhinav/seg_weights/fpn_epoch-34_loss-1.0681_val_loss-1.2505.h5"

    model.load_weights(weights_file, by_name=True, skip_mismatch=True)

    gt_seg_list = []
    pred_seg_list = []

    for i in range(len(val_orig_images_list)):
        image_file = val_orig_images_list[i]
        seg_file = val_seg_images_list[i]

        img = cv2.imread(image_file)
        seg_image = cv2.imread(seg_file, cv2.IMREAD_UNCHANGED)

        orig_img = img

        img = cv2.resize(img, (input_shape[0], input_shape[1]))
        img = img.astype(np.int16) - np.array(subtract_mean)
        # img = img_to_array(img) / 127.5 - 1

        seg_image = cv2.resize(seg_image, (input_shape[0], input_shape[1]), interpolation = cv2.INTER_NEAREST)

        pred = model.predict([[img]])
        output = np.argmax(pred[0], axis=2)

        # print (output.shape, seg_image.shape)

        gt_seg_list.append(seg_image)
        pred_seg_list.append(output)
        # eval_out = eval_semantic_segmentation(output[:, :, np.newaxis], seg_image[:, :, np.newaxis])
        #
        # iou = eval_out["iou"]
        # mean_iou = eval_out["miou"]
        # pixel_acc = eval_out["pixel_accuracy"]
        # class_acc = eval_out["class_accuracy"]
        # mean_class_acc = eval_out["mean_class_accuracy"]
        # print (iou)
        # print (mean_iou)
        # print (pixel_acc)
        # print (class_acc)
        # print (mean_class_acc)

        # rgb_output = np.zeros([input_shape[0], input_shape[1], 3])
        #
        # for i in range(input_shape[0]):
        #     for j in range(input_shape[1]):
        #             rgb_output[i][j] = out_palette[output[i][j]]
        #             # break
        #
        # print (np.unique(output), np.sum(output), output.shape, output.dtype)
        # # output = np.uint8(output*16)
        # cv2.imshow("Input", orig_img)
        # cv2.imshow("Output", rgb_output)
        # cv2.waitKey(0)

        print ("i : ", i)

        # if(i > 20):
        #     break

    gt_seg_list = np.asarray(gt_seg_list)
    pred_seg_list = np.asarray(pred_seg_list)

    print (gt_seg_list.shape, pred_seg_list.shape)
    eval_out = eval_semantic_segmentation(pred_seg_list, gt_seg_list)

    iou = eval_out["iou"]
    mean_iou = eval_out["miou"]
    pixel_acc = eval_out["pixel_accuracy"]
    class_acc = eval_out["class_accuracy"]
    mean_class_acc = eval_out["mean_class_accuracy"]

    print ("IOU : ", iou)
    print ("Mean IOU : ", mean_iou)
    print ("Pixel Acc", pixel_acc)
    print ("Class Acc", class_acc)
    print ("Mean Class Acc", mean_class_acc)