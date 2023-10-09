# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:15:30 2020

@author: Khalid.ELASNAOUI
"""
import os


import numpy as np
from tqdm import tqdm
from PIL import Image

from IntensityNormalization_CLAHE import CLAHE,normalize
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("[INFO] loading dataset...")
def Dataset_loader(DIR,   sigmaX=10,):
    if not os.path.exists('data_CV2'):
        os.makedirs('data_CV2')

    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".jpg" or ftype == ".JPG" or ftype == ".jpeg":
            img = read(PATH)
            # print(os.path.splitext(IMAGE_NAME)[0])

            img = CLAHE(img)
            I = normalize(img)
            img = Image.fromarray(I.astype('uint8'), 'RGB')
            print(os.path.join(DIR, os.path.splitext(IMAGE_NAME)[0] + '.jpg'))
            img.save(os.path.join(DIR, os.path.splitext(IMAGE_NAME)[0] + '.jpg'))

path = 'data_CV2/train'


classe='Healthy'

normalize(np.array(Dataset_loader(''+path+'/'+classe+'/'  )))
