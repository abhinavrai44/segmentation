import keras
import tensorflow as tf
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, UpSampling2D
from keras.regularizers import l2
from keras.models import Model

def conv_block(conv_block_input, kernel_size, filters, stage, block, dilation_rate=1, multigrid=[1, 2, 1],
               strides=(2, 2), l2_regularization=0.0005, bn_axis=3):
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'
    filter_count_1, filter_count_2, filter_count_3 = filters

    if (dilation_rate > 1):
        strides = (1, 1)
    else:
        multigrid = [1, 1, 1]

    x = Conv2D(filter_count_1, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(l2_regularization), name=conv_name + '2a', use_bias=True)(conv_block_input)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_count_2, (kernel_size, kernel_size), dilation_rate=dilation_rate * multigrid[1], padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name=conv_name + '2b',
               use_bias=True)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_count_3, (1, 1), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(l2_regularization), name=conv_name + '2c', use_bias=True)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2c')(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(filter_count_3, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_regularization), name=conv_name + '1', use_bias=True)(conv_block_input)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x


def identity_block(conv_block_input, kernel_size, filters, stage, block, dilation_rate=1, multigrid=[1, 2, 1],
                   l2_regularization=0.0005, bn_axis=3):
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'
    filter_count_1, filter_count_2, filter_count_3 = filters

    if (dilation_rate < 2):
        multigrid = [1, 1, 1]

    x = Conv2D(filter_count_1, (1, 1), dilation_rate=dilation_rate * multigrid[0], padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name=conv_name + '2a',
               use_bias=True)(conv_block_input)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_count_2, (kernel_size, kernel_size), dilation_rate=dilation_rate * multigrid[1], padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name=conv_name + '2b',
               use_bias=True)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_count_3, (1, 1), dilation_rate=dilation_rate * multigrid[2], padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name=conv_name + '2c',
               use_bias=True)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name + '2c')(x)
    x = Activation('relu')(x)

    x = Add()([x, conv_block_input])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x

def Resnet50(image_input, dilation_rate=1, multigrid=[1, 1, 1], bn_axis=3, l2_regularization=0.0005):
    print ("Input Shape : ", image_input.shape)
    image_input = ZeroPadding2D((3, 3))(image_input)
    conv_1 = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_regularization), name='conv1', use_bias=True)(image_input)
    conv_1 = BatchNormalization(name='norm_1')(conv_1)
    conv_1 = Activation('relu', name='relu_conv1')(conv_1)
    conv_1_1 = conv_1
    print("Conv_1 Shape : ", conv_1.shape)
    conv_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool_1')(conv_1)
    print("Maxpool Shape : ", conv_1.shape)

    # CONV2_x
    conv2_1 = conv_block(conv_1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),
                         l2_regularization=l2_regularization, bn_axis=bn_axis)
    conv2_2 = identity_block(conv2_1, 3, [64, 64, 256], stage=2, block='b', l2_regularization=l2_regularization,
                             bn_axis=bn_axis)
    conv2_3 = identity_block(conv2_2, 3, [64, 64, 256], stage=2, block='c', l2_regularization=l2_regularization,
                             bn_axis=bn_axis)
    print("conv2_3 Shape : ", conv2_3.shape)

    # CONV3_x
    conv3_1 = conv_block(conv2_3, 3, [128, 128, 512], dilation_rate=1, multigrid=multigrid, stage=3, block='a',
                         strides=(2, 2), l2_regularization=l2_regularization, bn_axis=bn_axis)
    conv3_2 = identity_block(conv3_1, 3, [128, 128, 512], dilation_rate=1, multigrid=multigrid, stage=3, block='b',
                             l2_regularization=l2_regularization, bn_axis=bn_axis)
    conv3_3 = identity_block(conv3_2, 3, [128, 128, 512], dilation_rate=1, multigrid=multigrid, stage=3, block='c',
                             l2_regularization=l2_regularization, bn_axis=bn_axis)
    conv3_4 = identity_block(conv3_3, 3, [128, 128, 512], dilation_rate=1, multigrid=multigrid, stage=3, block='d',
                             l2_regularization=l2_regularization, bn_axis=bn_axis)
    print("conv3_4 Shape : ", conv3_4.shape)

    # CONV4_x
    conv4_1 = conv_block(conv3_4, 3, [256, 256, 1024], dilation_rate=1, multigrid=multigrid, stage=4, block='a',
                         strides=(2, 2), l2_regularization=l2_regularization, bn_axis=bn_axis)
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
    print("conv4_6 Shape : ", conv4_6.shape)

    # CONV5_x
    conv5_1 = conv_block(conv4_4, 3, [512, 512, 2048], dilation_rate=1, multigrid=multigrid, stage=5, block='a',
                         strides=(2, 2), l2_regularization=l2_regularization, bn_axis=bn_axis)
    conv5_2 = identity_block(conv5_1, 3, [512, 512, 2048], dilation_rate=1, multigrid=multigrid, stage=5, block='b',
                             l2_regularization=l2_regularization, bn_axis=bn_axis)
    conv5_3 = identity_block(conv5_2, 3, [512, 512, 2048], dilation_rate=1, multigrid=multigrid, stage=5, block='c',
                             l2_regularization=l2_regularization, bn_axis=bn_axis)
    print("conv5_3 Shape : ", conv5_3.shape)

    return conv_1_1, conv2_3, conv3_4, conv4_6, conv5_3

def smooth_block(input_tensor,filters,block_number,stri):
    input_tensor = Conv2D(filters, 1, strides=(1, 1), name='fpn_'+stri+'_conv_'+str(block_number))(input_tensor)
    input_tensor = BatchNormalization(axis=3, name='fpn_'+stri+'_bn_'+str(block_number))(input_tensor)
    input_tensor = Activation('relu')(input_tensor)
    return input_tensor


def Feature_pyramid_network(C1,C2,C3,C4,C5):
    P5 = UpSampling2D(size=(2,2),name='fpn_5_upsample')(C5)
    P5 = smooth_block(P5,1024,5,'pre_smooth')
    P5 = Add()([P5,C4])
    P5 = Activation('relu')(P5)
    P5 = smooth_block(P5,1024,5,'smooth')

    P4 = UpSampling2D(size=(2, 2), name='fpn_4_upsample')(P5)
    P4 = smooth_block(P4, 512, 4, 'pre_smooth')
    P4 = Add()([P4, C3])
    P4 = Activation('relu')(P4)
    P4 = smooth_block(P4, 512, 4, 'smooth')

    P3 = UpSampling2D(size=(2, 2), name='fpn_3_upsample')(P4)
    P3 = smooth_block(P3, 256, 3, 'pre_smooth')
    P3 = Add()([P3, C2])
    P3 = Activation('relu')(P3)
    P3 = smooth_block(P3, 256, 3, 'smooth')

    print ("C1 : ", C1.shape)
    P2 = UpSampling2D(size=(2, 2), name='fpn_2_upsample')(P3)
    P2 = smooth_block(P2, 64, 2, 'pre_smooth')
    P2 = Add()([P2, C1])
    P2 = Activation('relu')(P2)
    P2 = smooth_block(P2,64, 2, 'smooth')

    # return [P2,P3,P4,P5]
    return P2

def FpnNet(image_size=(512, 512, 3), n_classes='20', mode='training',
           l2_regularization=0.0005, output_stride=16, levels=[6, 3, 2, 1], upsample_type='bilinear'):
    if image_size[2] == 3:
        bn_axis = 3
    else:
        bn_axis = 1

    image_input = Input(shape=image_size)

    [C1, C2, C3, C4, C5] = Resnet50(image_input, bn_axis, multigrid=[1, 1, 1], l2_regularization=l2_regularization)

    fpn_out = Feature_pyramid_network(C1, C2, C3, C4, C5)

    fpn_out = Conv2D(n_classes, (3, 3), padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name='fpn_final',
               use_bias=True)(fpn_out)

    fpn_out = UpSampling2D(size=(2,2),name='fpn_final_upsample')(fpn_out)

    print ("FPN Shape : ", fpn_out.shape)

    out = Activation("softmax")(fpn_out)
    model = Model(inputs=image_input, outputs=out)

    return model

