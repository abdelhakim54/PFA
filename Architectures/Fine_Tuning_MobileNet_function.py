# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:18:35 2020

@author: Khalid.ELASNAOUI
"""



import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten




def Binary_MobileNetV2_FineTunning(input_shape, optimizer,ClassificationType,loss,NbClass) :

#imports the mobilenet model and discards the last 1000 neuron layer.
    pre_trained_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")


    

    x=pre_trained_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu',kernel_regularizer=l2(0.01))(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu',kernel_regularizer=l2(0.01))(x) #dense layer 2
    x=Dense(512,activation='relu',kernel_regularizer=l2(0.01))(x) #dense layer 3
    out = Dense(NbClass, activation=ClassificationType,name='output_layer',kernel_regularizer=l2(0.01))(x)
    New_model = Model(inputs=pre_trained_model.input, outputs=out)
    New_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return New_model


