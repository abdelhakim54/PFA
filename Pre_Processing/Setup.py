# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:03:54 2020

@author: Khalid.ELASNAOUI
"""


from tensorflow.keras.optimizers.legacy import Adam
def Confuguration():

#**************************************************************************
    path='data_preprocessed'
    
    NBepochs= 1
    batch_size = 32
    ImageW,ImageH=224,224
    
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
 #**************************************************************************
    #********BINARY CLASSIFICATION
    ClassificationType='sigmoid'
    loss = 'binary_crossentropy'
    class_mode='binary'
    NbClass=1
    #**************************************************************************   
    
    train_directory = ''+path+'/train/'
    test_directory = ''+path+'/validation/'
#**************************************************************************

    input_shape=(ImageW,ImageH,3)
    target_size = (ImageW, ImageH)
#**************************************************************************
    return NBepochs,batch_size,ImageW,ImageH,train_directory,test_directory,input_shape,target_size,optimizer,ClassificationType,loss,NbClass,class_mode
