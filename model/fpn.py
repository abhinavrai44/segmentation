import keras
import tensorflow as tf
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, concatenate
from keras.regularizers import l2
from keras.models import Model

from model.vgg16 import vgg16_encoder
from model.resnet import resnet_encoder
from model.mobilenet import mobilenet_encoder

def fpn(image_shape, num_classes, backbone = "resnet50"):
    if(backbone == "vgg16"):
        image_input, f1, f2, f3, f4, f5 = vgg16_encoder(image_shape = image_shape)
    if(backbone == "resnet50"):
        image_input, f1, f2, f3, f4, f5 = resnet_encoder(image_shape = image_shape, number_layers = 50)
    if(backbone == "resnet101"):
        image_input, f1, f2, f3, f4, f5 = resnet_encoder(image_shape = image_shape, number_layers = 101)
    if(backbone == "mobilenet"):
        image_input, f1, f2, f3, f4, f5 = mobilenet_encoder(image_shape = image_shape)

    print ("Building Feature Pyramid")
    P5 = f5
    P5 = Conv2D(256, (1, 1), activation='relu', padding='same')(P5)
    P5 = BatchNormalization()(P5)
    P5_ret = Conv2D(256, (3, 3), activation='relu', padding='same')(P5)
    P5_ret = BatchNormalization()(P5_ret)

    print(P5_ret.shape)

    P4 = f4
    P4 = Conv2D(256, (1, 1), activation='relu', padding='same')(P4)
    P4 = BatchNormalization()(P4)
    P5_upsample = UpSampling2D(size=(2,2), name='P5_upsample')(P5)
    P4 = Add()([P5_upsample, P4])
    P4_ret = P4
    P4_ret = Conv2D(256, (3, 3), activation='relu', padding='same')(P4)
    P4_ret = BatchNormalization()(P4_ret)
    
    print(P4_ret.shape)

    P3 = f3
    P3 = Conv2D(256, (1, 1), activation='relu', padding='same')(P3)
    P3 = BatchNormalization()(P3)
    P4_upsample = UpSampling2D(size=(2,2), name='P4_upsample')(P4)
    P3 = Add()([P4_upsample, P3])
    P3_ret = P3
    P3_ret = Conv2D(256, (3, 3), activation='relu', padding='same')(P3)
    P3_ret = BatchNormalization()(P3_ret)
    
    print(P3_ret.shape)

    P2 = f2
    P2 = Conv2D(256, (1, 1), activation='relu', padding='same')(P2)
    P2 = BatchNormalization()(P2)
    P3_upsample = UpSampling2D(size=(2,2), name='P3_upsample')(P3)
    P2 = Add()([P3_upsample, P2])
    P2_ret = P2
    P2_ret = Conv2D(256, (3, 3), activation='relu', padding='same')(P2)
    P2_ret = BatchNormalization()(P2_ret)

    print(P2_ret.shape)

    print ("Concatenation")
    P5_ret = Conv2D(128, (3, 3), activation='relu', padding='same')(P5_ret)
    P5_ret = BatchNormalization()(P5_ret)
    P5_ret = Conv2D(128, (3, 3), activation='relu', padding='same')(P5_ret)
    P5_ret = BatchNormalization()(P5_ret)
    P5_ret = UpSampling2D(size=(8,8), interpolation='bilinear')(P5_ret)

    print(P5_ret.shape)

    P4_ret = Conv2D(128, (3, 3), activation='relu', padding='same')(P4_ret)
    P4_ret = BatchNormalization()(P4_ret)
    P4_ret = Conv2D(128, (3, 3), activation='relu', padding='same')(P4_ret)
    P4_ret = BatchNormalization()(P4_ret)
    P4_ret = UpSampling2D(size=(4,4), interpolation='bilinear')(P4_ret)
    
    print(P4_ret.shape)

    P3_ret = Conv2D(128, (3, 3), activation='relu', padding='same')(P3_ret)
    P3_ret = BatchNormalization()(P3_ret)
    P3_ret = Conv2D(128, (3, 3), activation='relu', padding='same')(P3_ret)
    P3_ret = BatchNormalization()(P3_ret)
    P3_ret = UpSampling2D(size=(2,2), interpolation='bilinear')(P3_ret)

    print(P3_ret.shape)

    P2_ret = Conv2D(128, (3, 3), activation='relu', padding='same')(P2_ret)
    P2_ret = BatchNormalization()(P2_ret)
    P2_ret = Conv2D(128, (3, 3), activation='relu', padding='same')(P2_ret)
    P2_ret = BatchNormalization()(P2_ret)

    print(P2_ret.shape)

    P_concat = concatenate([P5_ret, P4_ret, P3_ret, P2_ret])
    P_concat = Conv2D(num_classes, (3, 3), activation='relu', padding='same')(P_concat)
    P_concat = BatchNormalization()(P_concat)

    print ("Final Concat : ", P_concat.shape)
    out = UpSampling2D(size=(4,4), interpolation='bilinear')(P_concat)
    out = Activation("softmax")(out)

    print ("Output : ", out.shape)

    model = Model(inputs=image_input, outputs=out)
    
    return model
