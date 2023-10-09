# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 09:50:22 2019

@author: Khalid.ELASNAOUI
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import time

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

#from keras.utils.vis_utils import plot_model
from Fine_Tuning_DenseNet201_Function import Binary_DenseNet201_FineTunning
from MyImageDataGenerator import myDataAugmentation
from UsefulFunctions import SaveHistory,PlotFigures,ConfutionMatrix_ClassificationReport


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from Setup import Confuguration
#**************************************************************************
start_time = time.time()
NBepochs,batch_size,ImageW,ImageH,train_directory,test_directory,input_shape,target_size,optimizer,ClassificationType,lossV,NbClass,ClassMode=Confuguration()

training_set,test_set=myDataAugmentation(train_directory,test_directory,batch_size,ImageW, ImageH,ClassMode)
#**************************************************************************
print("\n\t[INFO] DenseNet201_FineTunning...")
MyModel=Binary_DenseNet201_FineTunning(input_shape,optimizer,ClassificationType,lossV,NbClass)
MyModel.summary()
#**************************************************************************
H=list(MyModel.metrics_names)
#training
print("\n\t[INFO] Training...")
early_stopping = EarlyStopping(monitor='val_loss', patience=1)#or monitor='val_acc'
mc = ModelCheckpoint('Model/DensNet_FT.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5,
                                            min_lr=0.000001, cooldown=2)
steps_per_epoch = (training_set.classes.shape[0]//batch_size)
validation_steps = (test_set.classes.shape[0]//batch_size)


print("\nnum_of_train_samples  "+ str(steps_per_epoch))
print("num_of_test_samples  "+ str(validation_steps))


# #Fine Tuning
# pre_trained_modelLayers[481].name
# for layer in pre_trained_modelLayers[481:]:
    # layer.trainable = True


# MyModel.compile(loss=lossV,
              # optimizer=optimizer,
              # metrics=['acc'])




history = MyModel.fit_generator(training_set,
                         steps_per_epoch = steps_per_epoch,
                         epochs = NBepochs,
                         verbose = 1,
                         validation_data = test_set,
                         validation_steps = validation_steps,
                         #class_weight={0:1 , 1:10},
                         #callbacks = [early_stopping,mc]#interrupt training when the validation loss isn't decreasing anymore
                         callbacks = [mc,learning_rate_reduction] 
                         )
                                            
#**************************************************************************


#**************************************************************************
print("\n\t[INFO] TEST...")
print(history.history.keys())

SaveHistory(history,'DensNet_FT')
PlotFigures(history,MyModel,'DensNet_FT')
validation_steps = (test_set.classes.shape[0]//batch_size+1)#+1 Obligatoire pour la matrice de confusion
ConfutionMatrix_ClassificationReport(MyModel,'DensNet_FT',test_set,validation_steps)
print('Training time: %s' % (time.time() - start_time))
