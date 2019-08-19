import cv2
import numpy as np
import os
import glob
import sys

source_file = "/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Downloads/VOC_SSD/VOC2012/SegmentationClass/*.png"
dest_file = "/media/abhinav/8d21f7ab-e8c6-4e2e-b086-db78b777abf0/abhinav/Downloads/VOC_SSD/VOC2012/seg_train/"

count = 0

palette = {
	         (192, 224, 224) : 0 ,
	         (224, 224, 192) : 0 ,
			 (  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }

for filepath in glob.iglob(source_file):
    print(filepath)

    img_name = filepath.split("/")[-1]
    print (img_name)

    source_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    target_img = np.zeros((source_img.shape[0], source_img.shape[1]), dtype=np.int16)

    print (target_img.shape, source_img.shape)

    for i in range(source_img.shape[0]):
        for j in range(source_img.shape[1]):
            try:
                target_img[i, j] = palette[(source_img[i][j][2], source_img[i][j][1], source_img[i][j][0])]
            except:
                print(filepath, source_img[i][j][2], source_img[i][j][1], source_img[i][j][0])
                print()
                sys.exit(0)

    print(np.unique(target_img))
    # target_img = target_img.astype(int)

    target_img_path = dest_file + img_name

    cv2.imwrite(target_img_path, target_img)

    print (count)
    count = count + 1
    # cv2.imshow("Source", source_img)
    # cv2.imshow("Target", target_img)
    # cv2.waitKey(0)
    # z = 1
    # break