# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:28:37 2020

@author: User
"""


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

# import cv2
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

def dice_categorical_crossentropy(y_true, y_pred): # model.compile에 loss에 들어갈부분 (original model)
    res = 1 + 0.1*keras.losses.categorical_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
    return res
    #return 1 + 0.1*keras.losses.categorical_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1e-08):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1) # -1: 가장 나중의 차원
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

#--------------------------------------------------------------------------

# this function returns targeted class into background
def create_adv_label(data_gen, which_target, num_classes):
    for (img,label) in data_gen:
        label[label==which_target] = 0 # YJS added
        return label
    
def get_y_true(data_gen):
    for (img,label) in data_gen:
        return label

def get_x(data_gen):
    for (img,label) in data_gen:
        return img    


def fgsm_generate(which_target, num_classes, data_gen, model):
    y = create_adv_label(data_gen, which_target, num_classes)
    x = get_x(data_gen)
    x = tf.convert_to_tensor(x)
    # def fgsm(x, y):
        # 1. clip min, max
        # x = model.input
    asserts = []
    
    # Added 2020-02-20: Getting logits =======
    # softmax_layer = model.get_layer(index=-1)
    # logit = softmax_layer.input
    y_pred = model(x)
    # y_pred = model.predict(x)
    #===================================
    
    # y_pred = model(x)
        # if clip_min is not None:
        #     asserts.append(utils_tf.assert_greater_equal(x, tf.cast(clip_min, x.dtype)))
        
        # if clip_max is not None:
        #     asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))
    
    # y = y / reduce_sum(y, 1, keepdims=True)
    # y = tf.convert_to_tensor(y)
    # y = tf.data.Dataset.from_generator(y, tf.float32)
    # y = y / tf.reduce_sum(y, 1, keepdims=True)    
        # Compute loss
    loss = keras.losses.categorical_crossentropy(y, y_pred)
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit) 
    # loss = dice_categorical_crossentropy(y, logit)
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit, dim=-1)
    # 그냥 평범한 crossentropy로
    if which_target is not None:
    # if targeted:
        loss = -loss # 부호를 바꿔줌 
            
        # Define gradient of loss wrt input
            
    grad, = tf.gradients(loss, x)
        
        # if clip_grad:
        #     grad = utils_tf.zero_out_clipped_grads(grad, x, clip_min, clip_max) # 여기 좀더 수정
        
        # optimal_perturbation = optimize_linear(grad, eps, ord)
    optimal_perturbation = tf.sign(grad)
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
        
        # Add perturbation to original example to obtain adversarial example
        # adv_x = x + optimal_perturbation       
        # return adv_x
    return x + optimal_perturbation

    
def adv_customLoss(data_gen, model, which_target, num_classes):
    x_adv = fgsm_generate(which_target, num_classes,  data_gen, model)
    
    x_adv = tf.stop_gradient(x_adv)
    y_pred = model(x_adv)
    y_true = get_y_true(data_gen)
    
    def adv_dice_categorical_crossentropy(y_true, y_pred):
        return 1 + 0.1*keras.losses.categorical_crossentropy(y_true, y_pred) - adv_dice_coef(y_true, y_pred)
        
    def adv_dice_coef(y_true, y_pred, smooth=1e-08):
        y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y_true[..., 1:])
        y_pred_f = K.flatten(y_pred[..., 1:])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1) # -1: 가장 나중의 차원
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
        return K.mean((2. * intersection + smooth) / (union + smooth))
    def adv_dice_coef_1(y_true, y_pred, smooth=1e-08):
        y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y_true[..., 1:])
        y_pred_f = K.flatten(y_pred[..., 1:])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1) # -1: 가장 나중의 차원
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)        
        return K.mean((2. * intersection + smooth) / (union + smooth))
    def adv_dice_coef_2(y_true, y_pred, smooth=1e-08):            
        y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y_true[..., 2])
        y_pred_f = K.flatten(y_pred[..., 2])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
        return K.mean((2. * intersection + smooth) / (union + smooth))
    def adv_dice_coef_3(y_true, y_pred, smooth=1e-08):
        y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y_true[..., 3])
        y_pred_f = K.flatten(y_pred[..., 3])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
        return K.mean((2. * intersection + smooth) / (union + smooth))                
    return adv_dice_categorical_crossentropy, adv_dice_coef, adv_dice_coef_1, adv_dice_coef_2, adv_dice_coef_3

