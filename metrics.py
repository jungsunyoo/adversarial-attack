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

def model_argmax(sess, x, predictions, samples, feed=None):
    """
    Helper function that computes the current class prediction
    """
    feed_dict = {x: samples}
    if feed is not None: 
        feed_dict.update(feed)
    probabilities = sess.run(predictions, feed_dict)
    
    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis =1)
    
def lbfgs_objective(adv_x, targets, oimgs, sess, loss_, grad_, CONST):
    """ returns the function value and the gradient for fmin_1_bfgs_b"""
    loss = sess.run(loss_, feed_dict = {x: adv_x.reshape(oimgs.shape),
                                           targeted_label: targets, 
                                           ori_img: oimgs, 
                                           const: CONST
                                           })
    grad = sess.run(grad_, feed_dict = {x: adv_x.reshape(oimgs.shape), 
                                           targeted_label: targets, 
                                           ori_img: oimgs, 
                                           const: CONST
                                           })
    return loss, grad.flatten().astype(float)    
    

def lbfgs_generate(which_target, num_classes, data_gen, model, sess, **kwargs):
    
    clip_min = kwargs["clip_min"]
    clip_max = kwargs["clip_max"]
    batch_size = kwargs["batch_size"]
    binary_search_steps = kwargs["binary_search_steps"]
    initial_const = kwargs["initial_const"]
    max_iterations = kwargs["max_iterations"]
    
    
    
    # locals().update(kwargs)
    # score = dice_categorical_crossentropy
    x_np = get_x(data_gen)
    x = tf.convert_to_tensor(x_np)
    y_pred = model(x)
    targets = create_adv_label(data_gen, which_target, num_classes)
    y_true = create_adv_label(data_gen, which_target, num_classes) #targeted label 
    loss_ = dice_categorical_crossentropy(y_true, y_pred)
    # l2dist = 일단은 l2dist, const 없는 버전으로
    grad_, = tf.gradients(loss_, x)
    # clip_min=0 # kwargs
    # clip_max = 1 # kwargs 
    oimgs = np.clip(x_np, clip_min, clip_max)
    # initial_const = 1e-2 # kwargs
    CONST = np.ones(batch_size) * initial_const # should receive batch size
    
    # set the lower and upper bounds accordingly
    lower_bound = np.zeros(batch_size)
    upper_bound = np.ones(batch_size) * 1e10
    
    # set the box constraints for the optimization function
    clip_min = clip_min * np.ones(oimgs.shape[:])
    clip_max = clip_max * np.ones(oimgs.shape[:])
    clip_bound = list(zip(clip_min.flatten(), clip_max.flatten()))
    
    # placeholders for the best l2 and instance attack found so far
    o_bestl2 = [1e10] * batch_size
    o_bestattack = np.copy(oimgs)
    
    
    
    
    # OPTIMIZATION FUNCTION
    repeat = binary_search_steps >= 10
    for outer_step in range(binary_search_steps): # binary_search_steps도 kwargs로 빼기
        if repeat and outer_step == binary_search_steps - 1: # repeat도 kwargs
            CONST = upper_bound
        adv_x, _, __ = fmin_l_bfgs_b(
                lbfgs_objective, 
                oimgs.flatten().astype(float), 
                args = (targets, oimgs, sess, loss_, grad_, CONST), 
                bounds = clip_bound, 
                maxiter = max_iterations, 
                iprint=0)
        adv_x = adv_x.reshape(oimgs.shape)
        
        assert np.amax(adv_x) <= clip_max and np.amin(adv_x) >= clip_min, 'returns are invalid' # Make sure adv_x is within range
            
        # adjust the best result (i.e., the adversarial example with the smallest perturbation in terms of L_2 norm) found so far
        preds = np.atleast_1d(
            model_argmax(sess, x, logits, adv_x)
            )
        # _logger.debug("predicted labels are %s", preds)
        
        l2s = np.zeros(batch_size)
        for i in range(batch_size):
            l2s[i] = np.sum(np.square(adv_x[i] - oimgs[i]))
        
        for e, (l2, pred, ii) in enumerate(zip(l2s, preds, adv_x)):
            if l2 < o_bestl2[e] and attack_success(pred, np.argmax(targets[e]), targeted_attack):
                o_bestl2[e] = l2
                o_bestattack[e] = ii
                
        # adjust the constant as needed
        
        for e in range(batch_size):
            if attack_success(pred[e], np.argmax(targets[e]), 
                              targeted_attack):
          # success, divide const by two
              upper_bound[e] = min(upper_bound[e], CONST[e])
              if upper_bound[e] < 1e9:
                CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
            else:
              # failure, either multiply by 10 if no solution found yet
              #          or do binary search with the known upper bound
              lower_bound[e] = max(lower_bound[e], CONST[e])
              if upper_bound[e] < 1e9:
                CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
              else:
                CONST[e] *= 10        
                
        o_bestl2 = np.array(o_bestl2)
        mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
        # return the best solution found
    o_bestl2 = np.array(o_bestl2)
    return o_bestattack
    
        
    
    
    
    
def adv_customLoss(data_gen, model, which_target, num_classes, sess, **kwargs):
    # x_adv = fgsm_generate(which_target, num_classes,  data_gen, model)
    tf_dtype = tf.as_dtype('float32')
    softmax_layer = model.get_layer(index=-1)
    logit = softmax_layer.input
    targeted_attack = True
    y_target = create_adv_label(data_gen, which_target, num_classes)
    clip_min = kwargs["clip_min"]
    clip_max = kwargs["clip_max"]
    batch_size = kwargs["batch_size"]
    binary_search_steps = kwargs["binary_search_steps"]
    initial_const = kwargs["initial_const"]
    max_iterations = kwargs["max_iterations"]
    x = get_x(data_gen)
    x = tf.convert_to_tensor(x)
    # x = model.inputs
    
    attack = LBFGS_impl(sess, x, logit, y_target, targeted_attack, binary_search_steps, max_iterations,initial_const, clip_min, clip_max, num_classes, batch_size)
    # x_adv = lbfgs_generate(which_target, num_classes, data_gen, model, sess, **lbfgs_params)
    
    def lbfgs_wrap(x_val, y_val):
      """
      Wrapper creating TensorFlow interface for use with py_func
      """
      return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)    

    wrap = tf.py_func(lbfgs_wrap, [x, y_target], tf_dtype)
    wrap.set_shape(x.get_shape())   
    
    
    x_adv = wrap
    
    
    # x_adv = attack.attack
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

class LBFGS_impl(object):
  """
  Return a tensor that constructs adversarial examples for the given
  input. Generate uses tf.py_func in order to operate over tensors.
  :param sess: a TF session.
  :param x: A tensor with the inputs.
  :param logits: A tensor with model's output logits.
  :param targeted_label: A tensor with the target labels.
  :param binary_search_steps: The number of times we perform binary
                              search to find the optimal tradeoff-
                              constant between norm of the purturbation
                              and cross-entropy loss of classification.
  :param max_iterations: The maximum number of iterations.
  :param initial_const: The initial tradeoff-constant to use to tune the
                        relative importance of size of the purturbation
                        and cross-entropy loss of the classification.
  :param clip_min: Minimum input component value
  :param clip_max: Maximum input component value
  :param num_labels: The number of classes in the model's output.
  :param batch_size: Number of attacks to run simultaneously.
  """

  def __init__(self, sess, x, logits, targeted_label, targeted_attack,
               binary_search_steps, max_iterations, initial_const, clip_min,
               clip_max, nb_classes, batch_size):
    self.sess = sess
    self.x = x
    self.logits = logits
    assert logits.op.type != 'Softmax'
    self.targeted_label = targeted_label
    self.targeted_attack = targeted_attack
    self.binary_search_steps = binary_search_steps
    self.max_iterations = max_iterations
    self.initial_const = initial_const
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.batch_size = batch_size
    tf_dtype = tf.as_dtype('float32')
    self.repeat = self.binary_search_steps >= 10
    self.shape = tuple([self.batch_size] +
                       list(self.x.get_shape().as_list()[1:]))
    self.ori_img = tf.Variable(
        np.zeros(self.shape), dtype=tf_dtype, name='ori_img')
    self.const = tf.Variable(
        np.zeros(self.batch_size), dtype=tf_dtype, name='const')

    # self.score = softmax_cross_entropy_with_logits(labels=self.targeted_label, logits=self.logits)
    # self.score = 
    # loss_ = dice_categorical_crossentropy(y_true, y_pred)
    
    
    
    
    
    # self.l2dist = reduce_sum(tf.square(self.x - self.ori_img))
    # small self.const will result small adversarial perturbation
    # targeted attack aims at minimize loss against target label
    # untargeted attack aims at maximize loss against True label
    # if self.targeted_attack:
    #   self.loss = reduce_sum(self.score * self.const) + self.l2dist
    # else:
    #   self.loss = -reduce_sum(self.score * self.const) + self.l2dist
    self.loss = dice_categorical_crossentropy(self.targeted_label, self.logits)
    self.grad, = tf.gradients(self.loss, self.x)

  def attack(self, x_val, targets):
    """
    Perform the attack on the given instance for the given targets.
    """

    def lbfgs_objective(adv_x, self, targets, oimgs, CONST):
      """ returns the function value and the gradient for fmin_l_bfgs_b """
      loss = self.sess.run(
          self.loss,
          feed_dict={
              self.x: adv_x.reshape(oimgs.shape),
              self.targeted_label: targets,
              self.ori_img: oimgs,
              self.const: CONST
          })
      grad = self.sess.run(
          self.grad,
          feed_dict={
              self.x: adv_x.reshape(oimgs.shape),
              self.targeted_label: targets,
              self.ori_img: oimgs,
              self.const: CONST
          })
      return loss, grad.flatten().astype(float)

    def attack_success(out, target, targeted_attack):
      """ returns attack result """
      if targeted_attack:
        return out == target
      else:
        return out != target

    # begin the main part for the attack
    from scipy.optimize import fmin_l_bfgs_b
    oimgs = np.clip(x_val, self.clip_min, self.clip_max)
    CONST = np.ones(self.batch_size) * self.initial_const

    # set the lower and upper bounds accordingly
    lower_bound = np.zeros(self.batch_size)
    upper_bound = np.ones(self.batch_size) * 1e10

    # set the box constraints for the optimization function
    clip_min = self.clip_min * np.ones(oimgs.shape[:])
    clip_max = self.clip_max * np.ones(oimgs.shape[:])
    clip_bound = list(zip(clip_min.flatten(), clip_max.flatten()))

    # placeholders for the best l2 and instance attack found so far
    o_bestl2 = [1e10] * self.batch_size
    o_bestattack = np.copy(oimgs)

    for outer_step in range(self.binary_search_steps):
      # _logger.debug("  Binary search step %s of %s",
      #               outer_step, self.binary_search_steps)

      # The last iteration (if we run many steps) repeat the search once.
      if self.repeat and outer_step == self.binary_search_steps - 1:
        CONST = upper_bound

      # optimization function
      adv_x, _, __ = fmin_l_bfgs_b(
          lbfgs_objective,
          oimgs.flatten().astype(float),
          args=(self, targets, oimgs, CONST),
          bounds=clip_bound,
          maxiter=self.max_iterations,
          iprint=0)

      adv_x = adv_x.reshape(oimgs.shape)
      assert np.amax(adv_x) <= self.clip_max and \
          np.amin(adv_x) >= self.clip_min, \
          'fmin_l_bfgs_b returns are invalid'

      # adjust the best result (i.e., the adversarial example with the
      # smallest perturbation in terms of L_2 norm) found so far
      preds = np.atleast_1d(
          utils_tf.model_argmax(self.sess, self.x, self.logits,
                                adv_x))
      _logger.debug("predicted labels are %s", preds)

      l2s = np.zeros(self.batch_size)
      for i in range(self.batch_size):
        l2s[i] = np.sum(np.square(adv_x[i] - oimgs[i]))

      for e, (l2, pred, ii) in enumerate(zip(l2s, preds, adv_x)):
        if l2 < o_bestl2[e] and attack_success(pred, np.argmax(targets[e]),
                                               self.targeted_attack):
          o_bestl2[e] = l2
          o_bestattack[e] = ii

      # adjust the constant as needed
      for e in range(self.batch_size):
        if attack_success(preds[e], np.argmax(targets[e]),
                          self.targeted_attack):
          # success, divide const by two
          upper_bound[e] = min(upper_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
        else:
          # failure, either multiply by 10 if no solution found yet
          #          or do binary search with the known upper bound
          lower_bound[e] = max(lower_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
          else:
            CONST[e] *= 10

      _logger.debug("  Successfully generated adversarial examples "
                    "on %s of %s instances.",
                    sum(upper_bound < 1e9), self.batch_size)
      o_bestl2 = np.array(o_bestl2)
      mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
      _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

    # return the best solution found
    o_bestl2 = np.array(o_bestl2)
    return o_bestattack
