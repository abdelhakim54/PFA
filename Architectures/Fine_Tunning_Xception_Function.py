# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:19:17 2020

@author: Khalid.ELASNAOUI
"""
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import backend as k
from tensorflow.keras import layers
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

def Binary_Xception_FineTunning(input_shape, optimizer,ClassificationType,loss,NbClass) :
#Pre-trained Xception Model loading
    pre_trained_model = Xception(input_shape=input_shape, include_top=False, weights="imagenet")

    x = pre_trained_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    
    
    predictions = Dense(NbClass, activation=ClassificationType,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    for layer in pre_trained_model.layers:
        layer.trainable = False    
    

    for i, layer in enumerate(model.layers):

        if i < 115:
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy']) 
    
   
    return model
