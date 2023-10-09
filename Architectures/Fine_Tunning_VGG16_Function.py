# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:00:24 2020

@author: Khalid.ELASNAOUI


"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers,Model
from tensorflow.keras.regularizers import l2

#Fine-tuning the top layers of a a pre-trained network
def Binary_VGG16_FineTunning(input_shape, optimizer,ClassificationType,loss,NbClass) :
    #Load in Pretrained VGG16 Model
    pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
    
    for layer in pre_trained_model.layers:
        print(layer.name)
        layer.trainable = False
    
    print(len(pre_trained_model.layers))
    
    
    last_layer = pre_trained_model.get_layer('block5_pool')
    print('last layer output shape:', last_layer.output_shape)
    last_output = last_layer.output
    #Define the Model
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
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model