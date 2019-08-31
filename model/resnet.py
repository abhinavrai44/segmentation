import keras
import tensorflow as tf
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, UpSampling2D
from keras.regularizers import l2
from keras.models import Model

def conv_block(conv_block_input, kernel_size, filters, stage, block, strides=(2, 2), dilation_rate = 1, l2_regularization=0.0005, bn_axis=3):
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'
    filter_count_1, filter_count_2, filter_count_3 = filters

    x = Conv2D(filter_count_1, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(l2_regularization), name=conv_name + '2a', use_bias=True)(conv_block_input)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_count_2, (kernel_size, kernel_size), padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate, kernel_regularizer=l2(l2_regularization), name=conv_name + '2b', use_bias=True)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_count_3, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name=conv_name + '2c', use_bias=True)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2c')(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(filter_count_3, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name=conv_name + '1', use_bias=True)(conv_block_input)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x


def identity_block(conv_block_input, kernel_size, filters, stage, block, dilation_rate = 1, l2_regularization=0.0005, bn_axis=3):
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'
    filter_count_1, filter_count_2, filter_count_3 = filters

    x = Conv2D(filter_count_1, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name=conv_name + '2a', use_bias=True)(conv_block_input)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_count_2, (kernel_size, kernel_size), padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate, kernel_regularizer=l2(l2_regularization), name=conv_name + '2b', use_bias=True)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_count_3, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name=conv_name + '2c', use_bias=True)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2c')(x)
    x = Activation('relu')(x)

    x = Add()([x, conv_block_input])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x

def resnet_encoder(image_shape, number_layers = 50, dilated_resnet = False):

    image_input = Input(shape=image_shape)

    bn_axis=3
    l2_regularization=0.0005

    print ("Input Shape : ", image_input.shape)
    image_input_pad = ZeroPadding2D((3, 3))(image_input)
    conv_1 = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_regularization), name='conv1', use_bias=True)(image_input_pad)
    conv_1 = BatchNormalization(name='norm_1')(conv_1)
    conv_1 = Activation('relu', name='relu_conv1')(conv_1)
    conv_1_1 = conv_1
    print("Conv_1 Shape : ", conv_1_1.shape)
    conv_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool_1')(conv_1)
    # print("Maxpool Shape : ", conv_1.shape)

    # CONV2_x
    conv2_1 = conv_block(conv_1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),
                         l2_regularization=l2_regularization, bn_axis=bn_axis)
    conv2_2 = identity_block(conv2_1, 3, [64, 64, 256], stage=2, block='b', l2_regularization=l2_regularization,
                             bn_axis=bn_axis)
    conv2_3 = identity_block(conv2_2, 3, [64, 64, 256], stage=2, block='c', l2_regularization=l2_regularization,
                             bn_axis=bn_axis)
    print("conv2_3 Shape : ", conv2_3.shape)

    # CONV3_x
    conv3_1 = conv_block(conv2_3, 3, [128, 128, 512], stage=3, block='a',
                         strides=(2, 2), l2_regularization=l2_regularization, bn_axis=bn_axis)
    conv3_2 = identity_block(conv3_1, 3, [128, 128, 512], stage=3, block='b',
                             l2_regularization=l2_regularization, bn_axis=bn_axis)
    conv3_3 = identity_block(conv3_2, 3, [128, 128, 512], stage=3, block='c',
                             l2_regularization=l2_regularization, bn_axis=bn_axis)
    conv3_4 = identity_block(conv3_3, 3, [128, 128, 512], stage=3, block='d',
                             l2_regularization=l2_regularization, bn_axis=bn_axis)
    print("conv3_4 Shape : ", conv3_4.shape)

    # CONV4_x
    if(dilated_resnet == False):
        conv4_1 = conv_block(conv3_4, 3, [256, 256, 1024], stage=4, block='a',
                             strides=(2, 2), l2_regularization=l2_regularization, bn_axis=bn_axis)
        if(number_layers == 101):
            for i in range(1,23):
                x = identity_block(conv4_1, 3, [256, 256, 1024], stage=4, block='b'+str(i), l2_regularization=l2_regularization,
                                     bn_axis=bn_axis)
            conv4_final = x
        else:
            conv4_2 = identity_block(conv4_1, 3, [256, 256, 1024], stage=4, block='b', l2_regularization=l2_regularization,
                                     bn_axis=bn_axis)
            conv4_3 = identity_block(conv4_2, 3, [256, 256, 1024], stage=4, block='c', l2_regularization=l2_regularization,
                                     bn_axis=bn_axis)
            conv4_4 = identity_block(conv4_3, 3, [256, 256, 1024], stage=4, block='d', l2_regularization=l2_regularization,
                                     bn_axis=bn_axis)
            conv4_5 = identity_block(conv4_4, 3, [256, 256, 1024], stage=4, block='e', l2_regularization=l2_regularization,
                                     bn_axis=bn_axis)
            conv4_6 = identity_block(conv4_5, 3, [256, 256, 1024], stage=4, block='f', l2_regularization=l2_regularization,
                                 bn_axis=bn_axis)
            conv4_final = conv4_6
    else:
        conv4_1 = conv_block(conv3_4, 3, [256, 256, 1024], stage=4, block='a',
                             strides=(1, 1), dilation_rate=2, l2_regularization=l2_regularization, bn_axis=bn_axis)
        if(number_layers == 101):
            for i in range(1,23):
                x = identity_block(conv4_1, 3, [256, 256, 1024], stage=4, block='b'+str(i), dilation_rate=2, l2_regularization=l2_regularization,
                                     bn_axis=bn_axis)
            conv4_final = x
        else:
            conv4_2 = identity_block(conv4_1, 3, [256, 256, 1024], stage=4, block='b', dilation_rate=2, l2_regularization=l2_regularization,
                                     bn_axis=bn_axis)
            conv4_3 = identity_block(conv4_2, 3, [256, 256, 1024], stage=4, block='c', dilation_rate=2, l2_regularization=l2_regularization,
                                     bn_axis=bn_axis)
            conv4_4 = identity_block(conv4_3, 3, [256, 256, 1024], stage=4, block='d', dilation_rate=2, l2_regularization=l2_regularization,
                                     bn_axis=bn_axis)
            conv4_5 = identity_block(conv4_4, 3, [256, 256, 1024], stage=4, block='e', dilation_rate=2, l2_regularization=l2_regularization,
                                     bn_axis=bn_axis)
            conv4_6 = identity_block(conv4_5, 3, [256, 256, 1024], stage=4, block='f', dilation_rate=2, l2_regularization=l2_regularization,
                                 bn_axis=bn_axis)
            conv4_final = conv4_6
    print("conv4_final Shape : ", conv4_final.shape)

    # CONV5_x
    if(dilated_resnet == False):
        conv5_1 = conv_block(conv4_final, 3, [512, 512, 2048], stage=5, block='a',
                             strides=(2, 2), l2_regularization=l2_regularization, bn_axis=bn_axis)
        conv5_2 = identity_block(conv5_1, 3, [512, 512, 2048], stage=5, block='b',
                                 l2_regularization=l2_regularization, bn_axis=bn_axis)
        conv5_3 = identity_block(conv5_2, 3, [512, 512, 2048], stage=5, block='c',
                                 l2_regularization=l2_regularization, bn_axis=bn_axis)
    else:
        conv5_1 = conv_block(conv4_final, 3, [512, 512, 2048], stage=5, block='a',
                             strides=(1, 1), dilation_rate=2, l2_regularization=l2_regularization, bn_axis=bn_axis)
        conv5_2 = identity_block(conv5_1, 3, [512, 512, 2048], stage=5, block='b', dilation_rate=2, 
                                 l2_regularization=l2_regularization, bn_axis=bn_axis)
        conv5_3 = identity_block(conv5_2, 3, [512, 512, 2048], stage=5, block='c', dilation_rate=2,
                                 l2_regularization=l2_regularization, bn_axis=bn_axis)
    print("conv5_3 Shape : ", conv5_3.shape)

    return image_input, conv_1_1, conv2_3, conv3_4, conv4_final, conv5_3