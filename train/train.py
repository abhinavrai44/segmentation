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

def compile(model):
    sgd = SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-4)
    model.compile(loss = "categorical_crossentropy", optimizer = adam, metrics = ["accuracy"])

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



