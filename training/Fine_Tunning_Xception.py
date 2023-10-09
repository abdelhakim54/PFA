# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:03:54 2020

@author: Khalid.ELASNAOUI
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import time

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau


from Fine_Tunning_Xception_Function import Binary_Xception_FineTunning
from MyImageDataGenerator import myDataAugmentation

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from UsefulFunctions import SaveHistory,PlotFigures,ConfutionMatrix_ClassificationReport
from Setup import Confuguration
#**************************************************************************

start_time = time.time()
NBepochs,batch_size,ImageW,ImageH,train_directory,test_directory,input_shape,target_size,optimizer,ClassificationType,lossV,NbClass,ClassMode=Confuguration()

training_set,test_set=myDataAugmentation(train_directory,test_directory,batch_size,ImageW, ImageH,ClassMode)
#**************************************************************************
print("\n\t[INFO] Xception_FineTunning...")
MyModel=Binary_Xception_FineTunning(input_shape,optimizer,ClassificationType,lossV,NbClass)
MyModel.summary()
#**************************************************************************
H=list(MyModel.metrics_names)
print(H)
#training
print("\n\t[INFO] Training...")
early_stopping = EarlyStopping(monitor='val_loss', patience=1)#or monitor='val_acc'
mc = ModelCheckpoint('Model/Xception_FT.h5', monitor='val_'+H[1], mode='max', verbose=1, save_best_only=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_'+H[1], patience=3, verbose=1, factor=0.5, 
                                            min_lr=0.000001, cooldown=2)

steps_per_epoch = (training_set.classes.shape[0]//batch_size)
validation_steps = (test_set.classes.shape[0]//batch_size)
#validation_steps = 800//batch_size+1#Obligatoire

print("\nnum_of_train_samples  "+ str(steps_per_epoch))
print("num_of_test_samples  "+ str(validation_steps))



history = MyModel.fit_generator(training_set,
                         steps_per_epoch = steps_per_epoch,
                         epochs = NBepochs,
                         verbose = 1,
                         validation_data = test_set,
                         validation_steps = validation_steps,
                          callbacks = [mc,learning_rate_reduction] 
                         )
                          
#**************************************************************************
                 
#**************************************************************************
print("\n\t[INFO] TEST...")
print(history.history.keys())

SaveHistory(history,'Xception_FT_')
PlotFigures(history,MyModel,'Xception_FT_')
validation_steps = (test_set.classes.shape[0]//batch_size+1)#+1 Obligatoire pour la matrice de confusion
ConfutionMatrix_ClassificationReport(MyModel,'Xception_FT_',test_set,validation_steps)
print('Training time: %s' % (time.time() - start_time))
