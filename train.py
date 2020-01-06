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
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import argparse

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
batch_size = 16
epochs = 50
learning_rate = 0.0001
num_classes = 4
base = 32
scale = 2

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

os.environ["CUDA_VISIBLE_DEVICES"]=device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

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
    model.compile(loss=dice_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate, decay=0.0),
                  metrics=[dice_coef, dice_coef_1, dice_coef_2, dice_coef_3])

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
                               'Scale': Scale})
    
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
