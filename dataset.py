from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#Tensorflow-gpu 1.14.0
#Keras 2.2.4
######################################################
## CHD                                              ##
######################################################
## "0": "Background"			                    ##
## "1": "LVM(Left Ventricle Myocardium)"	        ##
## "2": "LV(Left Ventricl Chamber)"                 ##
## "3": "RV(Right Ventricle Chamber)"          		##
######################################################

######################################################
## HCMP                                             ##
######################################################
## "0": "Background"			            ##
## "1": "LVM(Left Ventricle Myocardium)"	    ##
## "2": "APM(Anterior Papillary Muscle)"            ##
## "3": "PPM(Posteromedial Papillary Muscle)"	    ##
######################################################

import random
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import argparse

from model import *
from metrics import *
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras import backend as K

def data_generator(data,image_size, batch_size, num_classes, flag):
    seed = 103
    if flag == "train":
        data_gen_args_x = dict(
                             rescale=1/255.0,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             )
        data_gen_args_y = dict(
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             )
    else:      
        data_gen_args_x = dict(
            rescale=1/255.0
        )
        data_gen_args_y = dict()
        
    x_datagen = ImageDataGenerator(**data_gen_args_x)
    y_datagen = ImageDataGenerator(**data_gen_args_y)
    if flag == "train":
        x_gen = x_datagen.flow_from_directory(
            "./input/"+data+"/x_train/",
            class_mode = None,
            color_mode = "grayscale",
            target_size = (image_size, image_size),
            batch_size = batch_size,
            shuffle = True,
            seed = seed)
        y_gen = y_datagen.flow_from_directory(
            "./input/"+data+"/y_train/",
            class_mode = None,
            color_mode = "grayscale",
            target_size = (image_size, image_size),
            batch_size = batch_size,
            shuffle = True,
            seed = seed)
        
    elif flag == "valid":
        x_gen = x_datagen.flow_from_directory(
            "./input/"+data+"/x_valid/",
            class_mode = None,
            color_mode = "grayscale",
            target_size = (image_size, image_size),
            batch_size = batch_size,
            shuffle = True,
            seed = seed)
        y_gen = y_datagen.flow_from_directory(
            "./input/"+data+"/y_valid/",
            class_mode = None,
            color_mode = "grayscale",
            target_size = (image_size, image_size),
            batch_size = batch_size,
            shuffle = True,
            seed = seed)
        
    elif flag == "test":
        x_gen = x_datagen.flow_from_directory(
            "./input/"+data+"/x_test/",
            class_mode = None,
            color_mode = "grayscale",
            target_size = (image_size, image_size),
            batch_size = batch_size,
            shuffle = True,
            seed = seed)
        y_gen = y_datagen.flow_from_directory(
            "./input/"+data+"/y_test/",
            class_mode = None,
            color_mode = "grayscale",
            target_size = (image_size, image_size),
            batch_size = batch_size,
            shuffle = True,
            seed = seed)
    
    data_gen = zip(x_gen, y_gen)                
    for (img,label) in data_gen:
        label[label < 50] = 0
        label[(label >= 50) & (label < 112.5)] = 75
        label[(label >= 112.5) & (label < 187.5)] = 150
        label[label >= 187.5] = 225
        label[label==75] = 1
        label[label==150] = 2
        label[label==225] = 3
        label = to_categorical(label, num_classes=num_classes)
        yield (img,label)
