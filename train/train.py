from __future__ import division

import sys
sys.path.append("/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Desktop/nus_interactive_segmentaion/interactive-segmentation")

import six
import cv2
import keras
import tensorflow as tf
import numpy as np
import math
from model.fpn import FpnNet
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, TensorBoard, LearningRateScheduler
from generator.generator import data_generator_train, data_generator_val, aggregate_files
import keras.backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def mean_iou(y_true, y_pred):
  num_classes = 21
  score, update_op = tf.metrics.mean_iou(tf.argmax(y_true, axis=3), tf.argmax(y_pred, axis=3), num_classes)
  K.get_session().run(tf.local_variables_initializer())
  # K.get_session().run(tf.global_variables_initializer())
  with tf.control_dependencies([update_op]):
    final_score = tf.identity(score)
  return final_score

def FocalLoss(y_true, y_pred):
    gamma = 2
    alpha = 0.5
    y_pred = tf.maximum(y_pred, 1e-15)
    log_y_pred = tf.log(y_pred)
    focal_scale = tf.multiply(tf.pow(tf.subtract(1.0, y_pred), gamma), alpha)
    focal_loss = tf.multiply(y_true, tf.multiply(focal_scale, log_y_pred))
    return -tf.reduce_sum(focal_loss, axis=-1)

def lr_schedule(epoch):
    if epoch < 100:
        return 0.001
    elif (epoch < 300):
        return 0.0001
    else:
        return 0.00001

def compile(model):
    sgd = SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-4)

    # model.compile(loss = "categorical_crossentropy", optimizer = adam, metrics = ["accuracy"])
    model.compile(loss=FocalLoss, optimizer=adam, metrics=["accuracy"])

def train_model(model, input_shape, total_classes, train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list):
    compile(model)

    checkpoint_filepath = "/media/abhinav/Abhinav/seg_weights/fpn_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"
    csv_log_filepath = "/media/abhinav/Abhinav/seg_weights/csv/train.csv"
    tensorboard_filepath = "/media/abhinav/Abhinav/seg_weights/logs/"

    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=True)

    checkpoint = ModelCheckpoint(checkpoint_filepath, verbose=1, save_best_only=True, save_weights_only=True, mode='min')

    csv_logger = CSVLogger(filename=csv_log_filepath,
                           separator=',',
                           append=True)

    terminate_on_nan = TerminateOnNaN()

    tensorboard = TensorBoard(log_dir=tensorboard_filepath, histogram_freq=0,
                              write_graph=True, write_images=True)

    callbacks_list = [learning_rate_scheduler, checkpoint, csv_logger, terminate_on_nan, tensorboard]

    batch_size = 8
    train_size = len(train_orig_images_list)
    val_size = len(val_orig_images_list)

    initial_epoch = 0
    final_epoch = 60

    subtract_mean = [123, 117, 104]
    equalize = False
    brightness = (0.5, 2, 0.5)
    flip = 0.3
    translate = ((0,30), (0,30), 0.2)
    scale = (0.9, 1.1, 0.2)
    crop = (0.3, 0.7, 0.2)

    train_generator = data_generator_train(batch_size, input_shape, total_classes, train_orig_images_list, train_seg_images_list, subtract_mean, equalize, brightness, flip, translate, scale, crop)
    val_generator = data_generator_val(batch_size, input_shape, total_classes, val_orig_images_list, val_seg_images_list, subtract_mean, equalize)

    model.fit_generator(train_generator,
                        steps_per_epoch = math.ceil(train_size / batch_size),
                        epochs = final_epoch,
                        callbacks = callbacks_list,
                        initial_epoch = initial_epoch,
                        validation_data = val_generator,
                        validation_steps = math.ceil(val_size / batch_size)
                        )


if __name__ == '__main__':
    train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list = aggregate_files()


    input_shape = (256, 256, 3)
    total_classes = 21

    model = FpnNet(image_size = input_shape, n_classes = total_classes)

    weights_file = "/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Desktop/nus_interactive_segmentaion/segmentaion_code/weights/Resnet50.h5"

    model.load_weights(weights_file, by_name=True, skip_mismatch=True)
    train_model(model, input_shape, total_classes, train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list)



