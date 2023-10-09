# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:18:35 2020

@author: Khalid.ELASNAOUI
"""




from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten




def Binary_ResNet50_FineTunning(input_shape, optimizer,ClassificationType,loss,NbClass) :

#Pre-trained ResNet50 Model loading
    pre_trained_model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")


    pre_trained_model.summary()
    last_layer = pre_trained_model.output
# add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
    x = Dense(512, activation='relu',name='fc-1',kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu',name='fc-2',kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
# a softmax layer for 4 classes
    out = Dense(NbClass, activation=ClassificationType,name='output_layer',kernel_regularizer=l2(0.01))(x)

# this is the model we will train
    New_model = Model(inputs=pre_trained_model.input, outputs=out)   

    for layer in New_model.layers[:-6]:
        layers.trainable = False

    New_model.layers[-1].trainable

    New_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])    
    return New_model


