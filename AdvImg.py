# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:26:28 2020

@author: Jungsun Yoo
"""

import random
# import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import argparse

import pdb

from model import *
from metrics import *
from dataset import *
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras import backend as K





#1. Create dataset


test_name = "0x0x-test-x"
device = "0"
model_type = "U"
data = "chd"
option = "train"
image_size = 256
batch_size = 1
epochs = 50
learning_rate = 0.0001
num_classes = 4
base = 32
scale = 2

data_gen_args_x = dict(rescale=1/255.0,
                       width_shift_range=0.2,
                       height_shift_range=0.2,
                       zoom_range=0.2,
                      )
data_gen_args_y = dict(width_shift_range=0.2,
                       height_shift_range=0.2,
                       zoom_range=0.2,
                      )
seed = 103
x_datagen = ImageDataGenerator(**data_gen_args_x)
y_datagen = ImageDataGenerator(**data_gen_args_y)


x_gen = x_datagen.flow_from_directory(
            "./input/"+data+"/x_train/",
            class_mode = None,
            color_mode = "grayscale",
            target_size = (image_size, image_size),
            batch_size = batch_size,
            shuffle = True,
            save_to_dir = "./input/save_to_dir/x_train",
            seed = seed)
y_gen = y_datagen.flow_from_directory(
            "./input/"+data+"/y_train/",
            class_mode = None,
            color_mode = "grayscale",
            target_size = (image_size, image_size),
            batch_size = batch_size,
            shuffle = True,
            save_to_dir = "./input/save_to_dir/y_train",
            seed = seed)
# data_gen = zip(x_gen, y_gen) 

for label in y_gen[0]:          
    label[label < 50] = 0
    label[(label >= 50) & (label < 112.5)] = 75
    label[(label >= 112.5) & (label < 187.5)] = 150
    label[label >= 187.5] = 225
    label[label==75] = 1
    label[label==150] = 2
    label[label==225] = 3
    label = to_categorical(label, num_classes=num_classes)
y_true = label
xx = x_gen[0]


# 2. Load Pretrained Model 

pretrained_model = load_model(filepath="./save/"+str(test_name)+"_"+data+".hdf5",
              custom_objects={'dice_categorical_crossentropy': dice_categorical_crossentropy,
                              'dice_coef': dice_coef,
                              'dice_coef_1': dice_coef_1,
                              'dice_coef_2': dice_coef_2,
                              'dice_coef_3': dice_coef_3})

# 3. Getting signed_grad (based on https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)


#first, preprocess x (input) so that it becomes tensor
#use tf.expand_dims if necessary (if shape is not [1,image_size, image_size, 1] or [1, image_size, image_size, 4])
xxx = tf.convert_to_tensor(x_true)
yyy = tf.convert_to_tensor(y_true)
# xxx_np = tf.expand_dims(xx,0).eval(session=sess)
# yyy_np = tf.expand_dims(y_true,0).eval(session=sess)
xxx_np = xx
yyy_np = y_true 


loss_object = keras.losses.categorical_crossentropy(y_true, y_pred)


x = tf.placeholder("float", shape=[1, image_size, image_size, 1])
y = tf.placeholder("float", shape=[1,image_size,image_size,4])

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    # prediction = pretrained_model(input_image)
    prediction = pretrained_model(input_image)
    loss = keras.losses.categorical_crossentropy(input_label, prediction)
    # loss = loss_object(input_label, prediction)
    
  # Get the gradients of the loss w.r.t to the input image.

  gradient = tape.gradient(loss, input_image)
  sess = tf.InteractiveSession()
  sess.run(tf.initialize_all_variables())
  np_grad = sess.run(gradient, feed_dict = {x:xxx_np, y:yyy_np})
  
  # plt.imshow(np.squeeze(np_grad))
  
  # Get the sign of the gradients to create the perturbation
  # signed_grad = tf.sign(gradient)
  signed_grad = np.sign(np_grad)
  return signed_grad
# imshow(np.squeeze(signed_grad))
  

# 4. Displaying adversarial images
  
epsilon = 0.01 # range of 0~1

adv_x = np.squeeze(xxx_np) + np.squeeze(signed_grad)*epsilon
plt.imshow(adv_x)

noise = np.squeeze(signed_grad)
plt.imshow(signed_grad)