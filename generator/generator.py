import numpy as np
import cv2
import math
import random
import keras
from keras.preprocessing.image import img_to_array

def crop_resize(image, seg_image, min = 0.3, max = 0.7):
    rows, cols, ch = image.shape

    begin_row = math.ceil(np.random.uniform(0, min) * rows)
    end_row = math.ceil(np.random.uniform(max, 1) * rows)

    begin_col = math.ceil(np.random.uniform(0, min) * cols)
    end_col = math.ceil(np.random.uniform(max, 1) * cols)

    crop_image = image[begin_row:end_row, begin_col:end_col, :]
    crop_seg_image = seg_image[begin_row:end_row, begin_col:end_col]

    image = cv2.resize(crop_image, (cols, rows))
    seg_image = cv2.resize(crop_seg_image, (cols, rows), interpolation = cv2.INTER_NEAREST)

    return image, seg_image

def scale_image(image, seg_image, min = 0.9, max = 1.1):
    '''
    Scale the input image by a random factor picked from a uniform distribution
    over [min, max].

    Returns:
        The scaled image, the associated warp matrix, and the scaling value.
    '''

    rows,cols,ch = image.shape

    scale = np.random.uniform(min, max)

    M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, scale)
    return cv2.warpAffine(image, M, (cols, rows)), cv2.warpAffine(seg_image, M, (cols, rows), flags=cv2.INTER_NEAREST), M, scale

def translate_image(image, seg_image, horizontal = (0,30), vertical = (0,30)):
    '''
    Randomly translate the input image horizontally and vertically.

    Arguments:
        image (array-like): The image to be translated.
        horizontal (int tuple, optinal): A 2-tuple `(min, max)` with the minimum
            and maximum horizontal translation. A random translation value will
            be picked from a uniform distribution over [min, max].
        vertical (int tuple, optional): Analog to `horizontal`.

    Returns:
        The translated image and the horzontal and vertical shift values.
    '''
    rows,cols,ch = image.shape

    x = np.random.randint(horizontal[0], horizontal[1]+1)
    y = np.random.randint(vertical[0], vertical[1]+1)
    x_shift = random.choice([-x, x])
    y_shift = random.choice([-y, y])

    M = np.float32([[1,0,x_shift],[0,1,y_shift]])
    return cv2.warpAffine(image, M, (cols, rows)), cv2.warpAffine(seg_image, M, (cols, rows), flags=cv2.INTER_NEAREST), x_shift, y_shift


def flip_image(image):
    '''
    Flip the input image horizontally.
    '''
    return cv2.flip(image, 1)

def vary_brightness(image, min = 0.5, max = 2.0):
    '''
    Randomly change the brightness of the input image.
    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    random_br = np.random.uniform(min,max)

    #To protect against overflow: Calculate a mask for all pixels
    #where adjustment of the brightness would exceed the maximum
    #brightness value and set the value to the maximum at those pixels.
    mask = hsv[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    hsv[:,:,2] = v_channel

    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

def histogram_eq(image):
    image1 = np.copy(image)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

    image1[:,:,2] = cv2.equalizeHist(image1[:,:,2])

    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2BGR)

    return image1


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

def data_generator_train(batch_size, input_shape, total_classes, train_orig_images_list, train_seg_images_list, subtract_mean = None, equalize = False, brightness = None, flip = None, translate = None, scale = None, crop = None):
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


            if equalize:
                orig_image = histogram_eq(orig_image)

            if brightness:
                p = np.random.uniform(0,1)
                if p >= (1-brightness[2]):
                    orig_image = vary_brightness(orig_image, min=brightness[0], max=brightness[1])

            if flip: # Performs flips along the vertical axis.
                p = np.random.uniform(0,1)
                if p >= (1-flip):
                    orig_image = flip_image(orig_image)
                    seg_image = flip_image(seg_image)

            if translate:
                p = np.random.uniform(0,1)
                if p >= (1-translate[2]):
                    # Translate the image
                    orig_image, seg_image, xshift, yshift = translate_image(orig_image, seg_image, translate[0], translate[1])

            if scale:
                p = np.random.uniform(0,1)
                if p >= (1-scale[2]):
                    # Rescale the image
                    orig_image, seg_image, M, scale_factor = scale_image(orig_image, seg_image, scale[0], scale[1])

            if crop:
                p = np.random.uniform(0,1)
                if p >= (1-crop[2]):
                    # Crop and Resize the image
                    orig_image, seg_image = crop_resize(orig_image, seg_image, crop[0], crop[1])
                    
            if subtract_mean:
                orig_image = orig_image.astype(np.int16) - np.array(subtract_mean)
            # orig_image = img_to_array(orig_image) / 127.5 - 1

            x.append(orig_image)
            y.append(seg_image)

        x = np.array(x)
        y = np.array(y)

        y = keras.utils.to_categorical(y, total_classes)

        inputs = [x, y]
        yield x, y


def data_generator_val(batch_size, input_shape, total_classes, val_orig_images_list, val_seg_images_list, subtract_mean = None, equalize = False):
    total_images = len(val_orig_images_list)

    start = 0
    while True:
        x = []
        y = []

        for i in range(batch_size):
            index = start + i

            orig_image = cv2.imread(val_orig_images_list[index], cv2.IMREAD_UNCHANGED)
            seg_image = cv2.imread(val_seg_images_list[index], cv2.IMREAD_UNCHANGED)

            orig_image = cv2.resize(orig_image, (input_shape[0], input_shape[1]))
            seg_image = cv2.resize(seg_image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_NEAREST)

            if subtract_mean:
                orig_image = orig_image.astype(np.int16) - np.array(subtract_mean)

            if equalize:
                orig_image = histogram_eq(orig_image)

            # orig_image = img_to_array(orig_image) / 127.5 - 1

            x.append(orig_image)
            y.append(seg_image)

        start = start + batch_size
        if((start + batch_size) > len(val_orig_images_list)):
            start = 0

        x = np.array(x)
        y = np.array(y)

        y = keras.utils.to_categorical(y, total_classes)

        inputs = [x, y]
        yield x, y