#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:53:16 2021

@author: navin
"""


import numpy as np 
import cv2
import os

from tensorflow.keras.utils import Sequence
import tensorflow as tf
from keras.utils import to_categorical

TARGET_WIDTH, TARGET_HEIGHT = 176, 176


class InputDataLoader(Sequence):
    def __init__(self, input_image_path, output_label_path):
        self.input_image_path = input_image_path
        self.output_label_path = output_label_path
        self.input_image_list = os.listdir(self.input_image_path)
        self.output_label_list = os.listdir(self.output_label_path)
        self.input_image_list_order = sorted(self.input_image_list, key=lambda x:(x.split('_')[-1]))
        self.output_label_list_order = sorted(self.output_label_list, key=lambda x:(x.split('_')[-1]))
#        print( self.output_label_list_order )
    
    def get_image_batch(self, start_number, end_number, data_type):
        image_holder = []
        for each_image in range(start_number, end_number):
            if data_type == 'input':
                image = cv2.imread(os.path.join(self.input_image_path, self.input_image_list_order[each_image]))
#                print(self.input_image_path, self.input_image_list_order[each_image])
                image = image[2:178, 2:178]
                image = image/255.
            if data_type == 'output':
                image = cv2.imread(os.path.join(self.output_label_path, self.output_label_list_order[each_image]),0)
#                print(self.output_label_path, self.output_label_list_order[each_image])
                image = image[2:178, 2:178]
                image = image/255.
                image = to_categorical(image)
            image_holder.append(image)
#        image_holder = np.concatenate(image_holder, axis=-1)
        image_holder = np.array(image_holder)
        image_holder = image_holder.astype('float32')
        return image_holder
        
    def get_data(self, start, batch):
        image_input = self.get_image_batch(start, batch + start, 'input')
        label_output =  self.get_image_batch(start, batch + start, 'output')
        image = tf.convert_to_tensor(np.array(image_input))
        optical = tf.convert_to_tensor(np.array(label_output))
        return (image, optical)
               
#%%
#'''Data Tester'''
#input_path = '/home/navin/Image1/'
#label_path = '/home/navin/Label1/'
#batch_size = 3
#start = 0
#data_loader = MultiInputGenerator(input_path, label_path )
#for i in range(0,8, batch_size):
#    x_frame, output = data_loader.get_data(i, batch_size)
#%%
