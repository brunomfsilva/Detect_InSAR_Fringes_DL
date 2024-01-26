from keras.applications import VGG16, InceptionV3, ResNet50V2, InceptionResNetV2, Xception, VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import os
import tensorflow as tf
from keras import backend as K
import dill
import focal_loss



##FOLDERS
train_dir= './Dataset_patches/wra/train'
val_dir= './Dataset_patches/wra/validation'
test_dir= './Dataset_patches/wra/test'


#MODEL
conv=model(weights='imagenet', include_top=False, input_shape= (256,256,3))

model = conv.output
model = layers.Flatten()(model)
model = layers.Dense(1024, activation='relu')(model)
model = layers.BatchNormalization()(model)
model = layers.Dense(512, activation='relu')(model)
model = layers.BatchNormalization()(model)
model = layers.Dense(256, activation='relu')(model)
model = layers.BatchNormalization()(model)
model = layers.Dense(128, activation='relu')(model)
model = layers.BatchNormalization()(model)
model = layers.Dense(64, activation='relu')(model)
model = layers.BatchNormalization()(model)
model = layers.Dense(32, activation='relu')(model)
model = layers.BatchNormalization()(model)
out = layers.Dense(1, activation='sigmoid')(model)

model = models.Model(inputs=conv.input, outputs=out)


conv.trainable = True
set_trainable = False

for layer in model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


train_datagen = ImageDataGenerator(
    rescale=1./255)    


#OPEN AND PREPARE DATASET
datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(  
    train_dir,
    target_size=(256,256),
    batch_size=20,
    class_mode='binary')

validation_generator = datagen.flow_from_directory(  
    val_dir,
    target_size=(256,256),
    batch_size=20,
    class_mode='binary')


#CALLBACK SAVE BEST EPOCH
callb= [keras.callbacks.ModelCheckpoint(filepath='./FL2_vgg19_wra.h5', 
                                        monitor='val_loss', 
                                        save_best_only=True)]


#TRAIN MODEL
model.compile(loss=focal_loss.binary_focal_loss(gamma=2., alpha=.25),
              optimizer=optimizers.Adam(lr=0.00001),
              metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=260,
    epochs=150,
    validation_data=validation_generator,
    validation_steps=60,
    callbacks=callb)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(21, len(acc) + 21) #change to 1
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
 
dict_h=history.history

###########################
#EVALUATION
############################
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(256,256),
    batch_size=20,
    class_mode='binary')


model=load_model('./vgg19.h5', 
                 custom_objects={'binary_focal_loss_fixed': focal_loss.binary_focal_loss()} 
                 )

teste= model.evaluate(test_generator)


