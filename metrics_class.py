# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:08:25 2020

@author: User
"""


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

from scipy.optimize import fmin_l_bfgs_b


class LBFGS(Attack): # 자식 클래스를 선언할때 소괄호로 부모클래스 포함; 자식 = LBFGS, 부모 = Attack
    """
    LBFGS is the first adversarial attack for convolutional neural networks, 
    and is a target & iterative attack.
    : param model : cleverhans.model.Model
    : param sess: tf.Session
    : param dtypestr: dtype of the data
    : param kwargs: passed through to super constructor
    """
    def __init__(self, model, sess, dtypestr='float32', **kwargs)
    # 부모클래스: Attack? (Attack도 KerasModelWrapper(model)로 생성되니까?)
    # 정확히 말하면 class KerasModelWrapper(Model) ; Model은 class 아니고 def cnnmodel 로 생성 
        model = CallableModelWrapper
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


