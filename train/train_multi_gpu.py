from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"
import sys
sys.path.append("/home/abhinav/nus_interactive_segmentaion/interactive-segmentation")

import six
import cv2
import keras
import tensorflow as tf
import numpy as np
import math

from generator.generator import data_generator_train, data_generator_val, aggregate_files

from model.fpn import fpn
from model.fcn import fcn32, fcn8
from model.unet import modified_unet, original_unet
from model.pspnet import pspnet

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, TensorBoard, LearningRateScheduler
import keras.backend as K
from keras.utils import multi_gpu_model

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

from keras.models import Model
from keras.callbacks import ModelCheckpoint

class MultiGPUCheckpoint(ModelCheckpoint):
    
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model

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
    if epoch < 70:
        return 0.001
    elif (epoch < 140):
        return 0.0001
    elif (epoch < 200):
        return 0.00001
    else:
        return 0.000001

def compile(model):
    sgd = SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-4)

    model.compile(loss = "categorical_crossentropy", optimizer = adam, metrics = ["accuracy"])
    # model.compile(loss=FocalLoss, optimizer=adam, metrics=["accuracy"])

def train_model(model, input_shape, total_classes, train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list):
    compile(model)

    checkpoint_filepath = "/home/abhinav/seg_weights/fpn_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"
    csv_log_filepath = "/home/abhinav/seg_weights/csv/train.csv"
    tensorboard_filepath = "/home/abhinav/seg_weights/logs/"

    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=True)

    checkpoint = MultiGPUCheckpoint(checkpoint_filepath, verbose=1, save_best_only=True, save_weights_only=True, mode='min')

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
    final_epoch = 6000

    subtract_mean = [123, 117, 104]
    equalize = False
    brightness = (0.5, 2, 0.5)
    flip = 0.3
    translate = ((0,30), (0,30), 0.2)
    scale = (0.9, 1.1, 0.2)
    crop = (0.3, 0.7, 0.2)

    class_weights = np.asarray([0.0038620373161918067, 0.4258786462837951, 1.0, 0.3397074786078781, 0.5007799189042833, 0.499814129579366, 0.1733035897524969, 0.216985826449911, 0.11230052802548383, 0.26504260088197124, 0.3679469170396788, 0.22424409279306562, 0.17452693942923636, 0.3320480142433089, 0.26251479501952646, 0.06321442918648637, 0.45398601482332934, 0.336359922518134, 0.20992882787885261, 0.19032773644546333, 0.32279061056045905])

    train_generator = data_generator_train(batch_size, input_shape, total_classes, train_orig_images_list, train_seg_images_list, subtract_mean, equalize, brightness, flip, translate, scale, crop)
    val_generator = data_generator_val(batch_size, input_shape, total_classes, val_orig_images_list, val_seg_images_list, subtract_mean, equalize)

    model.fit_generator(train_generator,
                        steps_per_epoch = math.ceil(train_size / batch_size),
                        epochs = final_epoch,
                        callbacks = callbacks_list,
                        initial_epoch = initial_epoch,
                        validation_data = val_generator,
                        validation_steps = math.ceil(val_size / batch_size),
                        class_weight = class_weights
                        )


if __name__ == '__main__':
    train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list = aggregate_files()


    # input_shape = (256, 256, 3)
    input_shape = (512, 512, 3)
    total_classes = 21

    # model = fcn32(image_shape = input_shape, num_classes = total_classes, backbone = "vgg16")
    # model = fcn8(image_shape = input_shape, num_classes = total_classes, backbone = "vgg16")
    model = modified_unet(image_shape = input_shape, num_classes = total_classes, backbone = "resnet50")
    # model = original_unet(image_shape = input_shape, num_classes = total_classes)
    # model = pspnet(image_shape = input_shape, num_classes = total_classes, backbone = "resnet50")
    
    # model = fpn(image_shape = input_shape, num_classes = total_classes, backbone = "resnet50")
    model = multi_gpu_model(model, gpus=2)
    print (model.summary())
    weights_file = "/home/abhinav/seg_weights/without_class_weights/fpn_epoch-404_loss-0.8761_val_loss-0.8617.h5"
    # weights_file = "/media/abhinav/Abhinav/weights/mobilenet_1_0_224_tf.h5"
    # weights_file = "/media/abhinav/Abhinav/weights/VGG_ILSVRC_16_layers_fc_reduced.h5"
    # weights_file = "/media/abhinav/Abhinav/weights/ResNet-101-model.keras.h5"

    model.load_weights(weights_file, by_name=True, skip_mismatch=True)
    train_model(model, input_shape, total_classes, train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list)
