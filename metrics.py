from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#Tensorflow-gpu 1.14.0
#Keras 2.2.4
######################################################
## CHD                                              ##
######################################################
## "0": "Background"			            ##
## "1": "LVM(Left Ventricle Myocardium)"	    ##
## "2": "LV(Left Ventricl Chamber)"                 ##
## "3": "RV(Right Ventricle Chamber)"          	    ##
######################################################

######################################################
## HCMP                                             ##
######################################################
## "0": "Background"			            ##
## "1": "LVM(Left Ventricle Myocardium)"	    ##
## "2": "APM(Anterior Papillary Muscle)"            ##
## "3": "PPM(Posteromedial Papillary Muscle)"	    ##
######################################################

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import argparse

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras import backend as K

def dice_categorical_crossentropy(y_true, y_pred):
    return 1 + 0.1*keras.losses.categorical_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1e-08):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))

def dice_coef_1(y_true, y_pred, smooth=1e-08):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_true_f = K.flatten(y_true[..., 1])
    y_pred_f = K.flatten(y_pred[..., 1])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))

def dice_coef_2(y_true, y_pred, smooth=1e-08):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_true_f = K.flatten(y_true[..., 2])
    y_pred_f = K.flatten(y_pred[..., 2])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))

def dice_coef_3(y_true, y_pred, smooth=1e-08):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_true_f = K.flatten(y_true[..., 3])
    y_pred_f = K.flatten(y_pred[..., 3])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))
