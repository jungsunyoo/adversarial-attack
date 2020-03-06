from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#Tensorflow-gpu 1.14.0
#Keras 2.2.4
######################################################
## CHD                                              ##
######################################################
## "0": "Background"		                    ##
## "1": "LVM(Left Ventricle Myocardium)"            ##
## "2": "LV(Left Ventricl Chamber)"                 ##
## "3": "RV(Right Ventricle Chamber)"               ##
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

print(tf.__version__)
print(keras.__version__)

test_name = "0x0x-test-x"
device = "0"
model_type = "U"
data = "chd"
option = "train"
image_size = 256
batch_size = 5#16
epochs = 50
learning_rate = 0.0001
num_classes = 4
base = 32
scale = 2
which_target = 1

parser = argparse.ArgumentParser()
parser.add_argument('--test_name', type=str, default=test_name)
parser.add_argument('--device', type=str, default=device)
parser.add_argument('--model_type', type=str, default=model_type)
parser.add_argument('--data', type=str, default=data)
parser.add_argument('--option', type=str, default=option)
parser.add_argument('--image_size', type=int, default=image_size)
parser.add_argument('--batch_size', type=int, default=batch_size)
parser.add_argument('--epochs', type=int, default=epochs)
parser.add_argument('--learning_rate', type=float, default=learning_rate)
parser.add_argument('--num_classes', type=int, default=num_classes)
parser.add_argument('--base', type=int, default=base)
parser.add_argument('--scale', type=int, default=scale)
parser.add_argument('--which_target', type=int, default=which_target)
args = parser.parse_args()

test_name = args.test_name
device = args.device
model_type = args.model_type
data = args.data
option = args.option
image_size = args.image_size
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
num_classes = args.num_classes
base = args.base
scale = args.scale


lbfgs_params = {'clip_min': 0, 
                'clip_max': 1,
                'batch_size': 1, 
                'binary_search_steps': 5, 
                'initial_const': 1e-2,
                'max_iterations': 1000
                }


fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.} 
which_target = 1 # change label 2 to background

os.environ["CUDA_VISIBLE_DEVICES"]=device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Create TF session and set as Keras backend session
sess = tf.Session(config=config)
keras.backend.set_session(sess)


#y_sample = cv2.imread("./chd/y_train/sys_090_064.jpg")
#unique, counts = np.unique(y_sample, return_counts=True)
#print(dict(zip(unique, counts)))


train_gen = data_generator(data,image_size, batch_size, num_classes, flag='train')
valid_gen = data_generator(data,image_size, batch_size, num_classes, flag='valid')
test_gen = data_generator(data,image_size, batch_size, num_classes, flag='test')



if option == "train":
    input_img = Input(shape=(image_size, image_size, 1))
    if model_type == "U":
        model = U_Net(input_img, base, scale, num_classes)
    model.summary()
    
    def adv_decorator(data_gen, model, which_target, num_classes, sess, **kwargs):
        def adv_decorator2(func):   
            def wrapper(y_true, y_pred):
                tf_dtype = tf.as_dtype('float32')
                softmax_layer = model.get_layer(index=-1)
                logit = softmax_layer.input
                targeted_attack = True
                y_true = get_y_true(data_gen)
                y_target = create_adv_label(data_gen, which_target, num_classes)
                clip_min = kwargs["clip_min"]
                clip_max = kwargs["clip_max"]
                batch_size = kwargs["batch_size"]
                binary_search_steps = kwargs["binary_search_steps"]
                initial_const = kwargs["initial_const"]
                max_iterations = kwargs["max_iterations"]
                x = get_x(data_gen)
                x = tf.convert_to_tensor(x)

                attack = LBFGS_impl(sess, x, logit, y_target, targeted_attack, binary_search_steps, max_iterations,initial_const, clip_min, clip_max, num_classes, batch_size)
                
                # def lbfgs_wrap(x_val, y_val):
                #   """
                #   Wrapper creating TensorFlow interface for use with py_func
                #   """
                #   return np.array(attack.attack(x_val, y_val), dtype=tf.as_dtype('float32'))        
                wrap = tf.py_func(attack.attack, [x, y_target], tf_dtype)
                wrap.set_shape(x.get_shape())   
                x_adv = tf.stop_gradient(wrap)
                y_pred = model(x_adv)
                return func(y_true, y_pred)
            return wrapper
        return adv_decorator2
        
    @adv_decorator(data_gen=train_gen, model=model, which_target=which_target, num_classes=num_classes, sess=sess, **lbfgs_params)
    def adv_dice_categorical_crossentropy(y_true, y_pred):
        return 1 + 0.1*keras.losses.categorical_crossentropy(y_true, y_pred) - adv_dice_coef(y_true, y_pred)
    
    @adv_decorator(data_gen=train_gen, model=model, which_target=which_target, num_classes=num_classes, sess=sess, **lbfgs_params)
    def adv_dice_coef(y_true, y_pred, smooth=1e-08):
        y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y_true[..., 1:])
        y_pred_f = K.flatten(y_pred[..., 1:])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1) # -1: 가장 나중의 차원
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
        return K.mean((2. * intersection + smooth) / (union + smooth))

    @adv_decorator(data_gen=train_gen, model=model, which_target=which_target, num_classes=num_classes, sess=sess, **lbfgs_params)
    def adv_dice_coef_1(y_true, y_pred, smooth=1e-08):
        y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y_true[..., 1:])
        y_pred_f = K.flatten(y_pred[..., 1:])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1) # -1: 가장 나중의 차원
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)        
        return K.mean((2. * intersection + smooth) / (union + smooth))

    @adv_decorator(data_gen=train_gen, model=model, which_target=which_target, num_classes=num_classes, sess=sess, **lbfgs_params)
    def adv_dice_coef_2(y_true, y_pred, smooth=1e-08):            
        y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y_true[..., 2])
        y_pred_f = K.flatten(y_pred[..., 2])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
        return K.mean((2. * intersection + smooth) / (union + smooth))

    @adv_decorator(data_gen=train_gen, model=model, which_target=which_target, num_classes=num_classes, sess=sess, **lbfgs_params)
    def adv_dice_coef_3(y_true, y_pred, smooth=1e-08):
        y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true_f = K.flatten(y_true[..., 3])
        y_pred_f = K.flatten(y_pred[..., 3])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
        return K.mean((2. * intersection + smooth) / (union + smooth))
    
    model.compile(loss=dice_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate, decay=0.0),
                  metrics=[dice_coef, dice_coef_1, dice_coef_2, dice_coef_3, 
                           adv_dice_coef,  adv_dice_coef_1, adv_dice_coef_2, adv_dice_coef_3])

    checkpointer = ModelCheckpoint(
        filepath="./save/"+str(test_name)+"_"+data+".hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        period=1,
        mode="auto")
    csv_logger = CSVLogger("./log/"+str(test_name)+"_"+data+".log")
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.8,
        patience=5, 
        min_lr=0,
        verbose=1)

    model.fit_generator(
        train_gen,
        epochs=epochs,
        shuffle=True,
        steps_per_epoch=56751/batch_size,
        validation_data=valid_gen,
        validation_steps=19316/batch_size,
        callbacks=[checkpointer, csv_logger, reduce_lr],
        verbose=1)

                    
    model = load_model(filepath="./save/"+str(test_name)+"_"+data+".hdf5",
               custom_objects={'dice_categorical_crossentropy': dice_categorical_crossentropy,
                               'dice_coef': dice_coef,
                               'dice_coef_1': dice_coef_1,
                               'dice_coef_2': dice_coef_2,
                               'dice_coef_3': dice_coef_3,
                               'Scale': scale})
    
    result = model.evaluate_generator(
        test_gen,
        steps=18536/batch_size,
        verbose=1)
    
    print("dice_all, dice_lvm, dice_lv, dice_rv: ",
          round(result[1],3), round(result[2],3), round(result[3],3), round(result[4],3))
    
elif option == "evaluate":
    model = load_model(filepath="./save/"+str(test_name)+"_"+data+".hdf5",
                   custom_objects={'dice_categorical_crossentropy': dice_categorical_crossentropy,
                                   'dice_coef': dice_coef,
                                   'dice_coef_1': dice_coef_1,
                                   'dice_coef_2': dice_coef_2,
                                   'dice_coef_3': dice_coef_3,
                                   'Scale': Scale})
    
    result = model.evaluate_generator(
        test_gen,
        steps=18536/batch_size,
        verbose=1)
    
    print("dice_all, dice_lvm, dice_lv, dice_rv: ",
          round(result[1],3), round(result[2],3), round(result[3],3), round(result[4],3))
# elif option == "adv_training":
    # Train model which might be more robust to adversarial training; 0.5 + 0.5
    