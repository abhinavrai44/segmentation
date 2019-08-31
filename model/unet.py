import keras
import tensorflow as tf
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, concatenate
from keras.regularizers import l2
from keras.models import Model

from model.vgg16 import vgg16_encoder
from model.resnet import resnet_encoder
from model.mobilenet import mobilenet_encoder

# Todo Resnet101 weight load skips layers

def modified_unet(image_shape, num_classes, backbone = "vgg16"):
    if(backbone == "vgg16"):
        image_input, f1, f2, f3, f4, f5 = vgg16_encoder(image_shape = image_shape)
        resnet_multiplier = 1
    if(backbone == "resnet50"):
        image_input, f1, f2, f3, f4, f5 = resnet_encoder(image_shape = image_shape, number_layers = 50)
        resnet_multiplier = 2
    if(backbone == "resnet101"):
        image_input, f1, f2, f3, f4, f5 = resnet_encoder(image_shape = image_shape, number_layers = 101)
        resnet_multiplier = 2
    if(backbone == "mobilenet"):
        image_input, f1, f2, f3, f4, f5 = mobilenet_encoder(image_shape = image_shape)
        resnet_multiplier = 1

    P4 = f4

    P4 = ZeroPadding2D((1, 1))(P4)
    P4 = Conv2D(512 * resnet_multiplier, (3, 3), activation='relu', padding='valid')(P4)
    P4 = BatchNormalization()(P4)
    P4 = Conv2D(512 * resnet_multiplier, (3, 3), activation='relu', padding='same')(P4)
    P4 = BatchNormalization()(P4)

    P4 = UpSampling2D((2, 2))(P4)
    
    P3 = f3
    P3 = concatenate([P4, P3])
    P3 = ZeroPadding2D((1, 1))(P3)
    P3 = Conv2D(256 * resnet_multiplier, (3, 3), activation='relu', padding='valid')(P3)
    P3 = BatchNormalization()(P3)
    P3 = Conv2D(256 * resnet_multiplier, (3, 3), activation='relu', padding='same')(P3)
    P3 = BatchNormalization()(P3)

    P3 = UpSampling2D((2, 2))(P3)
    
    P2 = f2
    P2 = concatenate([P3, P2])
    P2 = ZeroPadding2D((1, 1))(P2)
    P2 = Conv2D(128 * resnet_multiplier, (3, 3), activation='relu', padding='valid')(P2)
    P2 = BatchNormalization()(P2)
    P2 = Conv2D(128 * resnet_multiplier, (3, 3), activation='relu', padding='same')(P2)
    P2 = BatchNormalization()(P2)

    P2 = UpSampling2D((2, 2))(P2)
    
    P1 = f1
    P1 = concatenate([P2, P1])

    P1 = ZeroPadding2D((1,1))(P1)
    P1 = Conv2D(64 , (3, 3), activation='relu', padding='valid')(P1)
    P1 = BatchNormalization()(P1)
    P1 = Conv2D(64 , (3, 3), activation='relu', padding='same')(P1)
    P1 = BatchNormalization()(P1)

    P1 = UpSampling2D((2, 2))(P1)
    P1 = Conv2D(64 , (3, 3), activation='relu', padding='same')(P1)
    P1 = BatchNormalization()(P1)
    out = Conv2D(num_classes, (3, 3), padding='same')(P1)
    out = Activation('softmax')(out)

    # print ("Output : ", out.shape)
    
    model = Model(inputs=image_input, outputs=out)
    
    return model

def original_unet(image_shape, num_classes):
    image_input = Input(image_shape)
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(image_input)
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(num_classes, (1, 1))(conv9)
    out = Activation('softmax')(conv10)

    model = Model(input=image_input, output=out)

    return model