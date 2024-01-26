import tensorflow as tf
import keras
#%env SM_FRAMEWORK=tf.keras
import segmentation_models as sm
from keras.models import load_model
from keras import optimizers
from keras.preprocessing.image import img_to_array, load_img, array_to_img
import glob
import cv2
import os 
import numpy as np 
import matplotlib.pyplot as plt
import random
from keras import backend as K
import dill
import random
from keras.preprocessing.image import ImageDataGenerator
import segmentation_metrics
 

#FOLDERS
train_folder='./Dataset_patches/wra/train/earthquake'
validation_folder='./Dataset_patches/wra/validation/earthquake'
test_folder='./Dataset_patches/wra/test/earthquake'

train_mask_folder='./Dataset_patches/wra/masks_train'
validation_mask_folder='./Dataset_patches/wra/masks_validation'



def generator_img(folder):
    for i in os.listdir(folder):
        img_= os.path.join(folder, i)
        img_= load_img(img_)
        img_= img_to_array(img_)
        img_[:] /= 255
        yield(img_)
    
def generator_mask(folder):
    for i in os.listdir(folder):
        mask_= os.path.join(folder, i)
        mask_= cv2.imread(mask_, 0).astype(np.float32)  
        mask_[:]= mask_/255
        mask_=np.expand_dims(mask_, axis=2)
        yield(mask_)

def my_image_mask_generator(generator_img, generator_mask, batch_size):
        gen = list(zip(generator_img, generator_mask))
        imgs=[]
        masks=[]
        count=0
        while True:                
                for (img, mask) in gen:
                    imgs.append(img)
                    masks.append(mask)
                    if len(imgs)==batch_size:
                        imgs=np.asarray(imgs)
                        masks=np.asarray(masks)
                        
                        yield (imgs, masks)   
                       
                        imgs=[]
                        masks=[]
                    
                    else:
                        continue


#OPEN  TRAIN GENERATORS
images_folder=train_folder
masks_folder=train_mask_folder

train_img_generator=generator_img(images_folder)
train_mask_generator= generator_mask(masks_folder)

train_generator= my_image_mask_generator(train_img_generator, train_mask_generator, batch_size=16)


#OPEN  VALIDATION GENERATORS
images_val_folder=validation_folder
masks_val_folder=validation_mask_folder

val_img_generator=generator_img(images_val_folder)
val_mask_generator= generator_mask(masks_val_folder)

val_generator= my_image_mask_generator(val_img_generator, val_mask_generator, batch_size=16)


model= sm.Unet('vgg19',
               classes=1,
               encoder_weights='imagenet')

model.compile(optimizer=optimizers.Adam(lr=0.0001), 
              loss=sm.losses.bce_jaccard_loss, 
              metrics=[segmentation_metrics.dice_coef, segmentation_metrics.mean_iou,'acc'] )  


callb= [keras.callbacks.ModelCheckpoint(filepath='./seg_vgg19_wra.h5', 
                                        monitor='val_loss', 
                                        save_best_only=True),
       keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coef', factor=0.95,
                                        patience=3, verbose=1, mode='max',
                                         cooldown=1, min_lr=0.00001)]


history = model.fit(train_generator,
                    steps_per_epoch=125,
                    epochs=150,
                    validation_data=val_generator,
                    validation_steps=62,
                    callbacks=callb
                    )