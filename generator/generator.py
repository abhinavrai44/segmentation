import numpy as np
import cv2
import math
import random
import keras
from keras.preprocessing.image import img_to_array

def aggregate_files():
    print("Collecting File Names")
    train12_files_path = "/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Downloads/VOC_SSD/VOC2012/ImageSets/Segmentation/train.txt"
    val12_files_path = "/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Downloads/VOC_SSD/VOC2012/ImageSets/Segmentation/val.txt"

    train12_file_lines = open(train12_files_path, "r")
    train12_files = train12_file_lines.readlines()

    val12_file_lines = open(val12_files_path, "r")
    val12_files = val12_file_lines.readlines()

    train_orig_images_12_root = "/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Downloads/VOC_SSD/VOC2012/JPEGImages/"
    train_seg_images_12_root = "/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Downloads/VOC_SSD/VOC2012/seg_train/"

    train_orig_images_list = []
    train_seg_images_list = []
    for file_name in train12_files:
        train_orig_images_list.append(train_orig_images_12_root + file_name[:-1] + ".jpg")
        train_seg_images_list.append(train_seg_images_12_root + file_name[:-1] + ".png")

    val_orig_images_list = []
    val_seg_images_list = []
    for file_name in val12_files:
        val_orig_images_list.append(train_orig_images_12_root + file_name[:-1] + ".jpg")
        val_seg_images_list.append(train_seg_images_12_root + file_name[:-1] + ".png")

    return train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list

def data_generator_train(batch_size, input_shape, total_classes, train_orig_images_list, train_seg_images_list):
    total_images = len(train_orig_images_list)

    while True:
        x = []
        y = []

        for i in range(batch_size):
            index = random.randint(0, total_images - 1)

            orig_image = cv2.imread(train_orig_images_list[index], cv2.IMREAD_UNCHANGED)
            seg_image = cv2.imread(train_seg_images_list[index], cv2.IMREAD_UNCHANGED)

            orig_image = cv2.resize(orig_image, (input_shape[0], input_shape[1]))
            seg_image = cv2.resize(seg_image, (input_shape[0], input_shape[1]), interpolation = cv2.INTER_NEAREST)

            orig_image = img_to_array(orig_image) / 127.5 - 1

            x.append(orig_image)
            y.append(seg_image)

        x = np.array(x)
        y = np.array(y)
        y = keras.utils.to_categorical(y, total_classes)

        inputs = [x, y]
        yield x, y


def data_generator_val(batch_size, input_shape, total_classes, val_orig_images_list, val_seg_images_list):
    total_images = len(val_orig_images_list)

    while True:
        x = []
        y = []

        for i in range(batch_size):
            index = random.randint(0, total_images - 1)

            orig_image = cv2.imread(val_orig_images_list[index], cv2.IMREAD_UNCHANGED)
            seg_image = cv2.imread(val_seg_images_list[index], cv2.IMREAD_UNCHANGED)

            orig_image = cv2.resize(orig_image, (input_shape[0], input_shape[1]))
            seg_image = cv2.resize(seg_image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_NEAREST)

            orig_image = img_to_array(orig_image) / 127.5 - 1

            x.append(orig_image)
            y.append(seg_image)

        x = np.array(x)
        y = np.array(y)
        y = keras.utils.to_categorical(y, total_classes)

        inputs = [x, y]
        yield x, y