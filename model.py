# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import *
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.engine import Layer
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

def U_Net(input_img, base, scale, num_classes):
    upsample = upsample_simple

    conv1 = Conv2D(base, 3, activation=None, padding='same', kernel_initializer='he_normal')(input_img)
    # FILTER # = 32 (BASE), FILTER SIZE = 3
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)
    conv1 = Conv2D(base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D((scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation='relu')(conv2)
    conv2 = Conv2D((scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D((scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation='relu')(conv3)
    conv3 = Conv2D((scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv_c = Conv2D((scale * scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool3)
    conv_c = BatchNormalization()(conv_c)
    conv_c = Activation(activation='relu')(conv_c)
    conv_c = Conv2D((scale * scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv_c)
    conv_c = BatchNormalization()(conv_c)
    conv_c = Activation(activation='relu')(conv_c)

    pool4 = upsample((scale * scale) * base, (2, 2), strides=(2, 2), padding='same')(conv_c)
    merge = concatenate([conv3, pool4])
    conv4 = Conv2D((scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation='relu')(conv4)
    conv4 = Conv2D((scale * scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation='relu')(conv4)

    pool5 = upsample((scale) * base, (2, 2), strides=(2, 2), padding='same')(conv4)
    merge = concatenate([conv2, pool5])
    conv5 = Conv2D((scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation='relu')(conv5)
    conv5 = Conv2D((scale) * base, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation='relu')(conv5)

    pool6 = upsample(base, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge = concatenate([conv1, pool6])
    conv6 = Conv2D(base, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    #-----YJS Modified------
    out_= Conv2D(num_classes, 3, activation=None, padding='same')(conv6)
    out = Activation(activation='softmax')(out_)
    
    #------
    # out = Conv2D(num_classes, 3, activation='softmax', padding='same')(conv6)

    model = Model(inputs=input_img, outputs=out)
    return model

