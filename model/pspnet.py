from math import ceil
import keras
import tensorflow as tf
from keras.layers import Layer, Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, AveragePooling2D, Concatenate
from keras.regularizers import l2
from keras.models import Model
from keras import layers
from keras.backend import tf as ktf
import keras.backend as K

from model.resnet import resnet_encoder

class Upsampling(Layer):

	def __init__(self, new_size, **kwargs):
		self.new_size = new_size
		super(Upsampling, self).__init__(**kwargs)

	def build(self, input_shape):
		super(Upsampling, self).build(input_shape)

	def call(self, inputs, **kwargs):
		new_height, new_width = self.new_size
		resized = ktf.image.resize_images(inputs, [new_height, new_width],
										  align_corners=True)
		return resized

	def compute_output_shape(self, input_shape):
		return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

	def get_config(self):
		config = super(Upsampling, self).get_config()
		config['new_size'] = self.new_size
		return config

def psp_block(prev_layer, level, feature_map_shape, input_shape):
	if ((input_shape[0] == 512) and (input_shape[1] == 512)):
		kernel_strides_map = {1: [64, 64],
							  2: [32, 32],
							  3: [22, 21],
							  6: [11, 9]}  # TODO: Level 6: Kernel correct, but stride not exactly the same as Pytorch
	else:
		raise ValueError("Pooling parameters for input shape " + str(input_shape) + " are not defined.")


	names = [
		"class_psp_" + str(level) + "_conv",
		"class_psp_" + str(level) + "_bn"
	]
	kernel = (kernel_strides_map[level][0], kernel_strides_map[level][0])
	strides = (kernel_strides_map[level][1], kernel_strides_map[level][1])
	prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
	prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev_layer)
	prev_layer = BatchNormalization(axis=3, name=names[1])(prev_layer)
	prev_layer = Activation('relu')(prev_layer)
	prev_layer = Upsampling(feature_map_shape)(prev_layer)
	return prev_layer

def pspnet(image_shape, num_classes, backbone = "vgg16"):
	if(backbone == "resnet50"):
		print ("Resnet50")
		image_input, f1, f2, f3, f4, f5 = resnet_encoder(image_shape = image_shape, number_layers = 50, dilated_resnet = True)
	if(backbone == "resnet101"):
		image_input, f1, f2, f3, f4, f5 = resnet_encoder(image_shape = image_shape, number_layers = 101, dilated_resnet = True)

	feature_map_size = (int(ceil(image_shape[0] / 8.0)), int(ceil(image_shape[1] / 8.0)))

	interp_block1 = psp_block(f5, 1, feature_map_size, image_shape)
	interp_block2 = psp_block(f5, 2, feature_map_size, image_shape)
	interp_block3 = psp_block(f5, 3, feature_map_size, image_shape)
	interp_block6 = psp_block(f5, 6, feature_map_size, image_shape)

	f5 = Concatenate()([interp_block1,
						interp_block2,
						interp_block3,
						interp_block6,
						f5])

	x = Conv2D(512, (1, 1), strides=(1, 1), padding="same", name="class_psp_reduce_conv", use_bias=False)(f5)
	x = BatchNormalization(axis=3, name="class_psp_reduce_bn")(x)
	x = Activation('relu')(x)

	out = Conv2D(num_classes, (1, 1), strides=(1, 1), name="class_psp_final_conv")(x)
	out = Activation('softmax')(out)

	out = Upsampling((image_shape[0], image_shape[1]))(out)

	print ("Output Shape : ", out.shape)

	model = Model(inputs=image_input, outputs=out)
	
	return model
