import keras
import tensorflow as tf
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout
from keras.regularizers import l2
from keras.models import Model

from model.vgg16 import vgg16_encoder
from model.resnet import resnet_encoder
from model.mobilenet import mobilenet_encoder

def fcn32(image_shape, num_classes, backbone = "vgg16"):
	if(backbone == "vgg16"):
		image_input, f1, f2, f3, f4, f5 = vgg16_encoder(image_shape = image_shape)
	if(backbone == "resnet50"):
		image_input, f1, f2, f3, f4, f5 = resnet_encoder(image_shape = image_shape, number_layers = 50)
	if(backbone == "resnet101"):
		image_input, f1, f2, f3, f4, f5 = resnet_encoder(image_shape = image_shape, number_layers = 101)
	if(backbone == "mobilenet"):
		image_input, f1, f2, f3, f4, f5 = mobilenet_encoder(image_shape = image_shape)

	P5 = f5

	P5 = Conv2D(4096, (7, 7), activation='relu', padding='same')(P5)
	P5 = Dropout(0.5)(P5)
	P5 = Conv2D(4096, (1, 1), activation='relu', padding='same')(P5)
	P5 = Dropout(0.5)(P5)

	P5 = Conv2D(num_classes, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(P5)

	print ("Output", P5.shape)
	out = Conv2DTranspose(num_classes, kernel_size=(64,64) , strides=(32, 32), padding='same', use_bias=False)(P5)	
	out = Activation('softmax')(out)
	print ("Output", out.shape)
	
	model = Model(inputs=image_input, outputs=out)
	
	return model

def fcn8(num_classes, image_shape, backbone = "vgg16"):
	if(backbone == "vgg16"):
		image_input, f1, f2, f3, f4, f5 = vgg16_encoder(image_shape = image_shape)
	if(backbone == "resnet50"):
		image_input, f1, f2, f3, f4, f5 = resnet_encoder(image_shape = image_shape, number_layers = 50)
	if(backbone == "resnet101"):
		image_input, f1, f2, f3, f4, f5 = resnet_encoder(image_shape = image_shape, number_layers = 101)
	if(backbone == "mobilenet"):
		image_input, f1, f2, f3, f4, f5 = mobilenet_encoder(image_shape = image_shape)


	P5 = f5, 
	P5 = Conv2D(4096, (7, 7), activation='relu', padding='same')(P5)
	P5 = Dropout(0.5)(P5)
	P5 = Conv2D(4096 , (1, 1), activation='relu', padding='same')(P5)
	P5 = Dropout(0.5)(P5)

	P5 = Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(P5)
	P5 = Conv2DTranspose(num_classes, kernel_size=(4,4), strides=(2,2), padding='same', use_bias=False)(P5)

	P4 = f4
	P4 = Conv2D(num_classes, (1, 1) , kernel_initializer='he_normal')(P4)
	
	P4 = Add()([P5, P4])

	P4 = Conv2DTranspose(num_classes, kernel_size=(4,4), strides=(2,2), padding='same', use_bias=False)(P4)

	P3 = f3
	P3 = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal')(P3)
	
	P3  = Add()([P3, P4])

	out = Conv2DTranspose(num_classes, kernel_size=(16,16), strides=(8,8), padding='same', use_bias=False)(P3)
	out = Activation('softmax')(out)

	model = Model(inputs=image_input, outputs=out)
	
	return model