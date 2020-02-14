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
    # pdb.set_trace()
    return res
    #return 1 + 0.1*keras.losses.categorical_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1e-08):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    # yjs added: y_true는 label 차원이 한개밖에 없으므로 0: 으로밖에 인덱싱할 수 없다
    # 세 배 부풀리기 해야할듯 (dimension을 맞춰주기 위해)
    
    
    # YJS added on 2020-01-14: 일렬로 세 번 이어붙이면 된다 (tf.concat 이용)
    # 아니, 일단 flatten됐을 때 어떻게 일렬이 되는지부터 파악하기 -> 1,2,3,1,2,3,1,2,3이 아니라 1,1,1,2,2,2,3,3,3이 되는듯
    # 그래서 그냥 어차피 세배 늘려서 union해서 더하는거랑 dice_coef1+2+3하는거랑 같을듯
    
    
    # y_true_f = tf.concat([K.flatten(y_true[..., 0:]), K.flatten(y_true[..., 0:]), K.flatten(y_true[..., 0:])], 0)
    # y_true_f = K.flatten(y_true[..., 0:])
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1) # -1: 가장 나중의 차원
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    
    # d1 = dice_coef_1(y_true, y_pred, smooth=1e-08)
    # d2 = dice_coef_2(y_true, y_pred, smooth=1e-08)
    # d3 = dice_coef_3(y_true, y_pred, smooth=1e-08)
    
    # return d1+d2+d3
    
    return K.mean((2. * intersection + smooth) / (union + smooth))

def dice_coef_1(y_true, y_pred, smooth=1e-08):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    # y_true_f = K.flatten(y_true[..., 0:])
    y_true_f = K.flatten(y_true[..., 1])
    y_pred_f = K.flatten(y_pred[..., 1])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))

def dice_coef_2(y_true, y_pred, smooth=1e-08):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    # y_true_f = K.flatten(y_true[..., 0:])
    y_true_f = K.flatten(y_true[..., 2])
    y_pred_f = K.flatten(y_pred[..., 2])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))

def dice_coef_3(y_true, y_pred, smooth=1e-08):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    # y_true_f = K.flatten(y_true[..., 0:])
    y_true_f = K.flatten(y_true[..., 3])
    y_pred_f = K.flatten(y_pred[..., 3])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))

#--------------------------------------------------------------------------

# this function returns targeted class into background
def create_adv_label(data_gen, which_target, num_classes):
    for (img,label) in data_gen:
        label[label < 50] = 0
        label[(label >= 50) & (label < 112.5)] = 75
        label[(label >= 112.5) & (label < 187.5)] = 150
        label[label >= 187.5] = 225
        label[label==75] = 1
        label[label==150] = 2
        label[label==225] = 3
        label[label==which_target] = 0 # YJS added
        label = to_categorical(label, num_classes=num_classes)
        yield label
        
def create_adv_label2(data_gen, which_target, num_classes):
    for (img,label) in data_gen:
        label[label < 50] = 0
        label[(label >= 50) & (label < 112.5)] = 75
        label[(label >= 112.5) & (label < 187.5)] = 150
        label[label >= 187.5] = 225
        label[label==75] = 1
        label[label==150] = 2
        label[label==225] = 3
        label[label==which_target] = 0 # YJS added
        # label = to_categorical(label, num_classes=num_classes)
        return label
    
def get_y_true(data_gen, num_classes):
    for (img,label) in data_gen:
        label[label < 50] = 0
        label[(label >= 50) & (label < 112.5)] = 75
        label[(label >= 112.5) & (label < 187.5)] = 150
        label[label >= 187.5] = 225
        label[label==75] = 1
        label[label==150] = 2
        label[label==225] = 3
        # label[label==which_target] = 0 # YJS added
        label = to_categorical(label, num_classes=num_classes)
        return label

def get_x(data_gen):
    for (img,label) in data_gen:
        return img    


def fgsm_generate(which_target, num_classes, data_gen, model):
    # x = [model.input]
    y = create_adv_label2(data_gen, which_target, num_classes)
    # y_true = get_y_true(data_gen, num_classes)
    x = get_x(data_gen)
    x = tf.convert_to_tensor(x)
    # def fgsm(x, y):
        # 1. clip min, max
        # x = model.input
    asserts = []
    y_pred = model(x)
        # if clip_min is not None:
        #     asserts.append(utils_tf.assert_greater_equal(x, tf.cast(clip_min, x.dtype)))
        
        # if clip_max is not None:
        #     asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))
    
    # y = y / reduce_sum(y, 1, keepdims=True)
    # y = tf.convert_to_tensor(y)
    # y = tf.data.Dataset.from_generator(y, tf.float32)
    y = y / tf.reduce_sum(y, 1, keepdims=True)    
        # Compute loss
    # loss = softmax_cross_entropy_with_logits(labels=y, logits=model.get_logits) 
    loss = dice_categorical_crossentropy(y, y_pred)
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
    # return fgsm
    
def adv_dice_categorical_crossentropy(model, fgsm_params, num_classses, which_target, data_gen): # model2.compile에 loss에 들어갈부분 (adversarial model)
    # x = get_x(data_gen)
    # x = [model.input]
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent_orig = 1 + 0.1*keras.losses.categorical_crossentropy(y, preds) - dice_coef(y, preds)
        # Generate adversarial examples
        x_adv = fgsm_generate(which_target, num_classses, data_gen, model)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)
        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = 1 + 0.1*keras.losses.categorical_crossentropy(y, preds_adv) - adv_dice_coef(model, fgsm, fgsm_params)     
        return 0.5 * cross_ent_orig + 0.5 * cross_ent_adv
    
    return adv_loss

def adv_dice_coef(model, fgsm_params, which_target, num_classses,  data_gen, smooth=1e-08):
    # x = [model.input]
    # y_true = [model.get_logits]
    def adv_coef(y, _):
        x_adv = fgsm_generate(which_target, num_classses,  data_gen, model)
        x_adv = tf.stop_gradient(x_adv)            
        y_pred_adv = model(x_adv)
        #y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y[..., 1:])
        y_pred_f = K.flatten(y_pred_adv[..., 1:])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1) # -1: 가장 나중의 차원
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)        
        return K.mean((2. * intersection + smooth) / (union + smooth))
    return adv_coef

def adv_dice_coef_1(model, fgsm_params, which_target, num_classses,  data_gen, smooth=1e-08):
    # x = [model.input]
    def adv_coef_1(y, _):           
        x_adv = fgsm_generate(which_target, num_classses, data_gen, model)
        x_adv = tf.stop_gradient(x_adv)   
        y_pred_adv = model(x_adv)
        # y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y[..., 1])
        y_pred_f = K.flatten(y_pred_adv[..., 1])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
        return K.mean((2. * intersection + smooth) / (union + smooth))
    return adv_coef_1

def adv_dice_coef_2(model, fgsm_params, which_target, num_classses,  data_gen, smooth=1e-08):
    # x = [model.input]
    def adv_coef_2(y,_):    
        x_adv = fgsm_generate(which_target, num_classses, data_gen, model)
        x_adv = tf.stop_gradient(x_adv)        
        y_pred_adv = model(x_adv)
        # y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y[..., 2])
        y_pred_f = K.flatten(y_pred_adv[..., 2])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
        return K.mean((2. * intersection + smooth) / (union + smooth))
    return adv_coef_2

def adv_dice_coef_3(model, fgsm_params, which_target, num_classses, data_gen,  smooth=1e-08):
    # x = [model.input]
    def adv_coef_3(y,_):
        x_adv = fgsm_generate(which_target, num_classses, data_gen, model)
        x_adv = tf.stop_gradient(x_adv)        
        y_pred_adv = model(x_adv)
        # y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y[..., 3])
        y_pred_f = K.flatten(y_pred_adv[..., 3])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
        return K.mean((2. * intersection + smooth) / (union + smooth))
    return adv_coef_3