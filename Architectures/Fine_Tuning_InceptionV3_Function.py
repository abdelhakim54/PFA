# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:05:52 2020

@author: Khalid.ELASNAOUI
"""

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

def Binary_InceptionV3_FineTunning(input_shape, optimizer,ClassificationType,NbClass,loss) :
    pre_trained_model = InceptionV3(input_shape=input_shape,  include_top=False, weights="imagenet")
    for layer in pre_trained_model.layers:
        print(layer.name)
        if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
            layer.trainable = True
            K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
            K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        else:
            layer.trainable = False

    print(len(pre_trained_model.layers))
    last_layer = pre_trained_model.get_layer('mixed10')
    print('last layer output shape:', last_layer.output_shape)
    last_output = last_layer.output
# Flatten the output layer to 1 dimension
    x = layers.GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
# Add a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
    x = layers.Dense(NbClass, activation=ClassificationType,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

# Configure and compile the model

    model = Model(pre_trained_model.input, x)
    print("\n\t[INFO] Fine Tuning...")
#we chose to train the top 2 inception blocks
    for layer in pre_trained_model.layers[249:]:
        layer.trainable = True
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
    return model