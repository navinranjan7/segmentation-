#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 10:48:07 2021

@author: navin
"""

#import numpy as np
import os 
import cv2
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

#%%
class DataAugmentation:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        
    def augmentation(self, save_prefix_name, horizontal_flip=False, vertical_flip=False):
        datagen = ImageDataGenerator(
#             rotation_range = 40, 
#             width_shift_range = 0.2,
#             height_shift_range = 0.2,
#             rescale = 1./255,
#             shear_range = 0.2,
#             zoom_range = 0.2,
             horizontal_flip = horizontal_flip,
             vertical_flip = vertical_flip)
        original_data = os.listdir(self.data_path)
        for each in original_data:
            fn, ext = os.path.splitext(each)
            image = load_img(self.data_path + each)
            image_array = img_to_array(image)
            pic_array = image_array.reshape((1, ) + image_array.shape)
            count =0
            for batch in datagen.flow(pic_array, batch_size = 1, save_to_dir=self.save_path, save_prefix = save_prefix_name + fn, 
                                      save_format = 'jpeg'):
                count += 1
                if count == 1:
                    break
def augmnentation_image_name_correction(data_path, save_path):
    augmentation_image_list = os.listdir(data_path)
    count = 0
    for each in augmentation_image_list:
        image = cv2.imread(data_path + each)
        fn, ext = os.path.splitext(each)
        text_split = fn.split('_')
        filename = text_split[0]
        for each in text_split[1:-2]:
            filename += '_' + each
        cv2.imwrite(save_path + filename + '.jpeg', image)
        count += 1
#%%
"""Augmentation for Input Image (Scratch Image)"""
data_dir = '/home/navin/IMAGE/'
save_dir_aug = '/home/navin/augmented_data/Image1/'
save_dir_name_cor ='/home/navin/augmented_data/Image/'
augmenation_process= DataAugmentation(data_dir, save_dir_aug)
augmenation_process.augmentation('v_', vertical_flip=True)
augmenation_process.augmentation('h_', horizontal_flip=True)
name_correction = augmnentation_image_name_correction(save_dir_aug, save_dir_name_cor)
#%%
"""Augmentation for Dataset Label """
data_dir = '/home/navin/LABEL/'
save_dir_aug = '/home/navin/augmented_data/label1/'
save_dir_name_cor ='/home/navin/augmented_data/label/' 
augmenation_process= DataAugmentation(data_dir, save_dir_aug)
augmenation_process.augmentation('v_', vertical_flip=True)
augmenation_process.augmentation('h_', horizontal_flip=True)
name_correction = augmnentation_image_name_correction(save_dir_aug, save_dir_name_cor)
#%%
#data_dir1 = '/home/navin/ATI_project/augmented_data/Image1/'
#image_list1 = os.listdir(data_dir1)
#image_list1[0]
##for each image in augmentated_image_list:
##    image = load_img(self.)
#fn, ext = os.path.splitext(image_list1[0])
#text = fn.split('_')
#a = text[0]
#for each in text[1:-2]:
#    a += '_' + each
##    a += 
#print(a)
#text
#fn[0:-7]+'.jpeg'
#%%



