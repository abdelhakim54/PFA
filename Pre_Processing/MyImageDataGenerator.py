# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:17:39 2020

@author: Khalid.ELASNAOUI
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def myDataAugmentation(train_directory,test_directory,batch_size,ImageW, ImageH,classMode):
    train_datagen = ImageDataGenerator(rescale=1 / 255.0,
                                   rotation_range=90,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   #validation_split=0.2#split data to train and test
                                   )

    test_datagen = ImageDataGenerator(rescale=1 / 255.0)

#**************************************************************************
#load the training data
    training_set=train_datagen.flow_from_directory(train_directory,
                                                    target_size=(ImageW, ImageH),
                                                    batch_size=batch_size,
                                                    class_mode=classMode,
                                                   #save_to_dir= "data_CV/augmente"
                                                    )
#load the test data
    test_set=test_datagen.flow_from_directory(test_directory,
                                                        target_size=(ImageW, ImageH),
                                                        batch_size=batch_size,
                                                        class_mode=classMode,
                                                        shuffle=False
                                                        )
    return training_set,test_set