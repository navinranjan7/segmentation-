''' Image segmentation for Scratch in PCB

1. Model: UNet
2. Loss Function: i) Binary Cross-entropy 
                 ii) Dice Loss
3. Metrics: IoU (intersection over Union) 
            intersection of ground-truth and predicted images divided by union of both
4. Code Type: Model Subclassing API 

'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import os

# Image data loader code
from dataloader import InputDataLoader

# Path for input and label images 
input_path = '/home/navin/IMAGE/'
label_path = '/home/navin/LABEL/'

# Hyperparameter for Model Architecture
STRIDES_CONV = (1, 1)
STRIDES_UP = (2, 2)

ACTIVATION =tf.nn.relu
PADDING = 'same'

CONV_POOL_SIZE = (2, 2)
CONV_POOL_STRIDES = (2, 2)
KERNEL_SIZE_CONV = (3, 3)
KERNEL_SIZE_RNN = (3, 3)
KERNEL_SIZE_UP = (2, 2)
#%%
''' Loss functions '''
from keras.utils import to_categorical

loss_fn_BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
def intersection_over_union(y_true, y_pred):
    y_true = to_categorical(y_true)
    y_pred = to_categorical(y_pred)
    intersection0 = np.logical_and(y_true[...,0], y_pred[...,0])
    union0 = np.logical_or(y_true[...,0], y_pred[...,0])
    intersection1 = np.logical_and(y_true[...,1], y_pred[...,1])
    union1 = np.logical_or(y_true[...,1], y_pred[...,1])
    intersection = (intersection0 + intersection1)/2
    union = (union0 + union1)/2
    iou_all = np.sum(intersection)/np.sum(union)
    iou_scratch = np.sum(intersection1)/np.sum(union1)
    return iou_all, iou_scratch

def dice_loss(y_true, y_pred):
    numerator = 2. * tf.math.reduce_sum(tf.math.multiply(y_true, y_pred))
    denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred))
    dice = numerator/denominator
    return 1-dice
#%%
class Pooling(layers.Layer):
    ''' Pooling operation: Reduce the spatial-resolution by half along width and height'''
    def __init__(self, out_channel):
        super(Pooling, self).__init__()
        self.conv_pooling = layers.Conv2D(out_channel,
                                          kernel_size=CONV_POOL_SIZE,
                                          strides=CONV_POOL_STRIDES,
                                          padding='valid',
                                          activation=ACTIVATION,
                                          kernel_initializer='he_uniform')
        self.b_n = layers.BatchNormalization()
    def call(self, input_tensor):
        max_pool = self.conv_pooling(input_tensor)
#        print(f"MaxPool :", max_pool.shape)
        return max_pool
#%%
class CNNBlock(layers.Layer):
    ''' Generate a CNN block with one convolution, batch normalization '''
    def __init__(self, out_channels):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels,
                                  kernel_size=KERNEL_SIZE_CONV,
                                  strides=STRIDES_CONV,
                                  padding='same',
                                  activation = ACTIVATION,
                                  kernel_initializer='he_uniform')
        self.b_n = layers.BatchNormalization()
        
    def call(self, input_tensor, training=False):
        convolution = self.conv(input_tensor, training=training)
        batch_normalization = self.b_n(convolution)
#        print(f"Conv : ", batch_normalization.shape)
        return batch_normalization
#%%
class EncoderBlock(layers.Layer):
    ''' Two CNNBlock as one block 
        params: Channels : Number of filters for convloutional layers   
    '''
    def __init__(self, channels):
        super(EncoderBlock, self).__init__()
        self.cnn1 = CNNBlock(channels[0])
        self.cnn2 = CNNBlock(channels[1])

    def call(self, input_tensor, training=False):
        convolution_block = self.cnn1(input_tensor, training=training)
        convolution_block = self.cnn2(convolution_block, training=training)
        return convolution_block
#%%
class DecoderBlock(layers.Layer):
    '''Generate a decoder block with one Upsampling, Skip connection from Encoder block and 2 CNNBlock'''
    def __init__(self, channels):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = layers.Conv2DTranspose(channels[0],
                                                     kernel_size=KERNEL_SIZE_UP,
                                                     strides=STRIDES_UP,
                                                     padding='valid',
                                                     activation=ACTIVATION,
                                                     kernel_initializer='he_uniform')
        self.b_n = layers.BatchNormalization()
        self.concat1 = layers.Concatenate(axis=-1)
        self.cnn_block1 = CNNBlock(channels[1])
        self.cnn_block2 = CNNBlock(channels[2])
        
    def call(self, input_tensor, skip_conn, skip_connection=False, training=False):
        up_sampling = self.conv_transpose(input_tensor, training=training)
        up_sampling = self.b_n(up_sampling, training=training)
        if skip_connection:
            up_sampling = self.concat1([up_sampling, skip_conn])
        convolution_block = self.cnn_block1(up_sampling, training=training)
        convolution_block = self.cnn_block2(convolution_block, training=training)
        return convolution_block
#%%
class ImageSegmentationModel(keras.Model):
    '''Optical flow Recurrent Convolutional Autoencoder for predicting optical flow '''
    def __init__(self):
        super(ImageSegmentationModel, self).__init__()
        #Encoder
        self.block1 = EncoderBlock([32, 32])   
        self.pool1 = Pooling(64)
        
        self.block2 = EncoderBlock([64, 64])
        self.pool2 = Pooling(128)
        
        self.block3 = EncoderBlock([128, 128])
        self.pool3 = Pooling(256)
        
        self.block4 = EncoderBlock([256, 256])   
        self.pool4 = Pooling(512)
        
        self.block5 = EncoderBlock([512, 512])
        
        #Decoder
        self.decode_block4 = DecoderBlock([256, 256, 256])
        self.decode_block3 = DecoderBlock([128, 128, 128])
        self.decode_block2 = DecoderBlock([64, 64, 64])
        self.decode_block1 = DecoderBlock([32, 32, 32])
        
        # output layer with classification probability for 2 class:  background and scratch
        self.out_layer = layers.Conv2D(filters=2,
                                  kernel_size=KERNEL_SIZE_CONV,
                                  strides=STRIDES_CONV,
                                  padding='same',
                                  activation = 'softmax',
                                  kernel_initializer='he_uniform')
        
    def call(self, inputs, training=False):
        e_block1 = self.block1(inputs, training=training)
        e_pool1 = self.pool1(e_block1, training=training)
        ### optical Encoder Block 2
        e_block2 = self.block2(e_pool1, training=training)
        e_pool2 = self.pool2(e_block2, training=training)
        ### block Encoder Block 3
        e_block3 = self.block3(e_pool2, training=training)
        e_pool3 = self.pool3(e_block3, training=training)
        ### block Encoder Block 4
        e_block4 = self.block4(e_pool3, training=training)
        e_pool4 = self.pool4(e_block4, training=training)
        
        e_block5 = self.block5(e_pool4, training=training)
        
        d_block4 = self.decode_block4(e_block5, e_block4, skip_connection=True, training=training)
        ## block Decoder Block 3
        d_block3 = self.decode_block3(d_block4, e_block3, skip_connection=True, training=training)
        ### block Decoder Block 2
        d_block2 = self.decode_block2(d_block3, e_block2, skip_connection=True, training=training)
        ### block Decoder Block 1
        d_block1 = self.decode_block1(d_block2, e_block1, skip_connection=True, training=training)
        
        # output layer
        output = self.out_layer(d_block1, training=training)
        return(output)
#%%
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#%%
model = ImageSegmentationModel()    #Create Model instance 

''' Train the model using GradientTape '''
@tf.function
def train_main(input_image, output):
    with tf.GradientTape() as tape:
        logits = model(input_image, training=True)              #forward pass
        loss = loss_fn_BCE(output, logits)                     # Compute the loss 
    grads = tape.gradient(loss , model.trainable_variables)     # Estimate Gradients 
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) # Update optimizer
    return loss
#%%
data_loader = InputDataLoader(input_path, label_path) # Create instance to load data from directory

def train_epoch(model, epochs = 100, batch_size = 16):
    stop = 200 # Train data --> from 0 to 1800, remaining for Test data
    bec_epoch_loss_train = []
    bec_epoch_loss_test = []
#    bec_loss_train = []
#    bec_loss_test = []
    for epoch in range (0, epochs):
        ''' Load data from directory in batch=batch_size and call train_main function to train the model'''
        bec_loss_train = []
        bec_loss_test = []
        for start in range(0, stop, batch_size):
            input_image, output = data_loader.get_data(start, batch_size)   # get input and label image
            loss_value = train_main(input_image, output)
            bec_loss_train.append(loss_value)

        ''' Visually check the model performance by saving predicted image and ground-truth image side-by-side'''
        for k in range(200, 284 - batch_size):
            input_image_test, output_test = data_loader.get_data(k, batch_size) #load test data
            test_logits = model(input_image_test, training=False) # get predicted result for input without training the model
            loss_value_test = loss_fn_BCE(output_test, test_logits)
            bec_loss_test.append(loss_value_test)
            
        mean_bec_loss_train = np.mean(bec_loss_train)
        mean_bec_loss_test = np.mean(bec_loss_test)
        bec_epoch_loss_train.append(mean_bec_loss_train)
        bec_epoch_loss_test.append(mean_bec_loss_test)
        print(f"Epochs {epoch} Training Loss: {mean_bec_loss_train}  Testing Loss: {mean_bec_loss_test}")
#%%
def generate_result(model, batch_size):
    for k in range(200, 284 - batch_size):
        input_image_test, output_test = data_loader.get_data(k, batch_size) #load test data
        val_logits = model(input_image_test, training=False) # get predicted result for input without training the model
#        bec_loss_test.append(loss_value)
        
        predict = val_logits.numpy()
        out = predict[0,:,:,:]
        out = out.argmax(axis = -1)
        out = out.reshape(176,176,1)
        out = (out*255) 
        
        true1 = output_test.numpy()
        out1 = true1[0,:,:,:]
        out1 = out1.argmax(axis = -1)
        out1 = out1.reshape(176,176,1)
        out1 = (out1*255)
#            o = cv2.hconcat([out,out1])
        cv2.imwrite(f"/home/navin/ATI_project/RESULT/BCE_final/{k}.png",out)  # CAHNGE TO SAVE PATH 
        cv2.imwrite(f"/home/navin/ATI_project/RESULT/BCE_final_true/{k}.png",out1)
#%%
with tf.device('/gpu:3'):
    train_epoch(model)
    generate_result(model, 8)
    #%%
#'''result Evaluation'''   
result_dice = '/home/navin/ATI_project/RESULT/BCE_final/'
result_BCE = '/home/navin/ATI_project/RESULT/BCE_final_true/'
result_label = '/home/navin/ATI_project/Label/'
#
result_dice_list = os.listdir(result_dice)
result_BCE_list = os.listdir(result_BCE)
result_label_list = os.listdir(result_label)
result_dice_list.sort()
result_BCE_list.sort()
result_label_list.sort()
dice_all = []
dice_scratch = []
BCE_all = []
BCE_scratch = []
for i in range (0, len(result_label_list)):
#    print(result_dice + result_dice_list[i])
    i_dice = cv2.imread(result_dice + result_dice_list[i], 0)/255.
    i_BCE = cv2.imread(result_BCE + result_BCE_list[i], 0)/255.
    i_label = cv2.imread(result_label + result_label_list[i], 0)/255.
    dice_iou_all, dice_iou_scratch = intersection_over_union(i_dice, i_label)
    bce_iou_all, bce_iou_scratch = intersection_over_union(i_BCE, i_label)
    dice_all.append(dice_iou_all)
    dice_scratch.append(dice_iou_all)
    BCE_all.append(bce_iou_all)
    BCE_scratch.append(bce_iou_scratch)
#%% 
