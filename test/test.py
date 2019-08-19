import sys
sys.path.append("/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Desktop/nus_interactive_segmentaion/segmentaion_code")

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

if __name__ == '__main__':
    train_orig_images_list, train_seg_images_list, val_orig_images_list, val_seg_images_list = aggregate_files()


    input_shape = (256, 256, 3)
    total_classes = 21

    model = FpnNet(image_size = input_shape, n_classes = total_classes)

    weights_file = "/media/abhinav/Abhinav/seg_weights/fpn_epoch-42_loss-0.6146_val_loss-1.1421.h5"

    model.load_weights(weights_file, by_name=True, skip_mismatch=True)

    for image_file in val_orig_images_list:
        img = cv2.imread(image_file)
        orig_img = img
        img = cv2.resize(img, (input_shape[0], input_shape[1]))
        img = img_to_array(img) / 127.5 - 1

        pred = model.predict([[img]])
        output = np.argmax(pred[0], axis=2)

        rgb_output = np.zeros([input_shape[0], input_shape[1], 3])

        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                    rgb_output[i][j] = out_palette[output[i][j]]
                    # break

        print (np.unique(output), np.sum(output), output.shape, output.dtype)
        # output = np.uint8(output*16)
        cv2.imshow("Input", orig_img)
        cv2.imshow("Output", rgb_output)
        cv2.waitKey(0)
        # break
