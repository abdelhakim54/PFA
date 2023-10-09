# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:19:17 2020

@author: Khalid.ELASNAOUI
"""
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras import backend as k
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

def Binary_DenseNet201_FineTunning(input_shape, optimizer,ClassificationType,loss,NbClass) :
#Pre-trained DenseNet201 Model loading
    pre_trained_model = DenseNet201(input_shape=input_shape, include_top=False, weights="imagenet")

    for layer in pre_trained_model.layers:
        print(layer.name)
        if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
            layer.trainable = True
            k.eval(k.update(layer.moving_mean, k.zeros_like(layer.moving_mean)))
            k.eval(k.update(layer.moving_variance, k.zeros_like(layer.moving_variance)))
        else:
            layer.trainable = False

    print(len(pre_trained_model.layers))

    last_layer = pre_trained_model.get_layer('relu')
    print('last layer output shape:', last_layer.output_shape)
    last_output = last_layer.output

#Define the Model
# Flatten the output layer to 1 dimension
    x = layers.GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
# Add a dropout rate of 0.7
    x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
    x = layers.Dense(NbClass, activation=ClassificationType,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

# Configure and compile the model

    model = Model(pre_trained_model.input, x)
    #Fine Tuning
    pre_trained_model.layers[481].name
    for layer in pre_trained_model.layers[481:]:
        layer.trainable = True
  
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])  
    

    return model
