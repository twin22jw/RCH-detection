#!/usr/bin/env python
# coding: utf-8

"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)

Borrowed code and ideas from zhixuhao's unet: https://github.com/zhixuhao/unet and then modified
"""

# load modules 
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K

Img_Width    = 256
Img_Height   = 256
Img_Channels = 7 # Depending on input scenarios 
Num_Classes  = 2

inputs  = Input((Img_Height, Img_Width, Img_Channels))

c1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(inputs)
c1 = tf.keras.layers.BatchNormalization(axis=-1)(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1) # optional
c1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c1)
c1 = tf.keras.layers.BatchNormalization(axis=-1)(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)


c2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p1)
c2 = tf.keras.layers.BatchNormalization(axis=-1)(c2)
c2 = tf.keras.layers.Dropout(0.1)(c2) # optional
c2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c2)
c2 = tf.keras.layers.BatchNormalization(axis=-1)(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)


c3 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p2)
c3 = tf.keras.layers.BatchNormalization(axis=-1)(c3)
c3 = tf.keras.layers.Dropout(0.2)(c3) # optional
c3 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c3)
c3 = tf.keras.layers.BatchNormalization(axis=-1)(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)


c4 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p3)
c4 = tf.keras.layers.BatchNormalization(axis=-1)(c4)
c4 = tf.keras.layers.Dropout(0.2)(c4) # optional
c4 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c4)
c4 = tf.keras.layers.BatchNormalization(axis=-1)(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(c4)


c5 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p4)
c5 = tf.keras.layers.BatchNormalization(axis=-1)(c5)
c5 = tf.keras.layers.Dropout(0.3)(c5) # optional
c5 = tf.keras.layers.BatchNormalization(axis=-1)(c5)
c5 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c5)

u6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides = (2, 2), padding = 'same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])

c6 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u6)
c6 = tf.keras.layers.Dropout(0.2) (c6)
c6 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])

c7 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u7)
c7 = tf.keras.layers.Dropout(0.2) (c7)
c7 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])

c8 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u8)
c8 = tf.keras.layers.Dropout(0.1) (c8)
c8 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis = 3)

c9 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c9)

outputs = tf.keras.layers.Conv2D(2, (1, 1), activation = 'sigmoid')(c9) 
print(outputs.shape)

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

model   = Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [dice_coef])
