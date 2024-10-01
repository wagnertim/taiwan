#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:01:58 2021

@author: timw
"""

# Here is where we transfer machine learning to python.  This script takes a
# previously generated .mat file, and loads it up, does the basic QC like 
# you'd expect it to be done before, and then uses tensorflow to turn that
# into something useful.  We hope.

#  Define the key inputs

learnRate=3.0e-4
numEpochs=100
numSteps=10
batchSize=32
alfa = 0.1 # Alpha for leaky ReLU
b=5.0   #  These need to be floats, not integers.
c=2.0   #  Otherwise, it will crash.
stopPatience=10
fixSeeds = 1
inputDropout = 0.1
dropout = 0.2
apexDropout = 0.3
finalDropout = 0.1

trainSplit = 0.85
valSplit = 0.13

inputFile='/home/timw/input/multibox_stripped_rainy.mat'



print('Importing packages')

import scipy.io
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from shutil import copyfile
import tensorflow.keras.backend as K


#  Fix the seeds, if desired, for repetability

if fixSeeds == 0:
	from numpy.random import seed
	seed(1)
	from tensorflow import random
	random.set_seed(2)


#  Define a custom loss function to weight the higher values more effectively.

#def weighted_mse(y_true,y_pred):
#	yToTheC = tf.math.pow(y_true,c)
#	W = tf.exp(tf.multiply(b,yToTheC))
#	sq = tf.math.pow(y_pred - y_true,2)
#	return K.mean(tf.multiply(W, sq))

def weighted_mse(y_true,y_pred):
	yC = tf.math.pow(y_true,c)
	return K.mean( tf.multiply(
		tf.exp(tf.multiply(b, yC)),
		tf.square(tf.subtract(y_pred, y_true))))

print('Defining the model')

def make_unet(input_shape):
    ip = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(64, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(ip)
    c1 = layers.Conv2D(64, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(c1)
    c1 = layers.Dropout(inputDropout)(c1)
    m1 = layers.MaxPool2D()(c1)
    c2 = layers.Conv2D(128, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(m1)
    c2 = layers.Conv2D(128, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(c2)
    c2 = layers.Dropout(dropout)(c2)
    m2 = layers.MaxPool2D()(c2)
    c3 = layers.Conv2D(256, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(m2)
    c3 = layers.Conv2D(256, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(c3)
    c3 = layers.Dropout(dropout)(c3)
    m3 = layers.MaxPool2D()(c3)
    c4 = layers.Conv2D(512, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(m3)
    c4 = layers.Conv2D(512, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(c4)
    c4 = layers.Dropout(dropout)(c4)
    m4 = layers.MaxPool2D()(c4)
    c5 = layers.Conv2D(1024, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(m4)
    c5 = layers.Conv2D(1024, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(c5)
    c5 = layers.Dropout(apexDropout)(c5)
    u1 = layers.Conv2DTranspose(1024, 2, strides=2, padding='same')(c5) #crop1 = layers.Cropping2D(4)(c4)
    crop1 = c4
    conc1 = layers.Concatenate()([u1, crop1])
    c6 = layers.Conv2D(512, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(conc1)
    c6 = layers.Conv2D(512, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(c6)
    c6 = layers.Dropout(dropout)(c6)
    u2 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c6) #crop2 = layers.Cropping2D(16)(c3)
    crop2 = c3
    conc2 = layers.Concatenate()([u2, crop2])
    c7 = layers.Conv2D(256, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(conc2)
    c7 = layers.Conv2D(256, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(c7)
    c7 = layers.Dropout(dropout)(c7)
    u3 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c7) #crop3 = layers.Cropping2D(40)(c2)
    crop3 = c2
    conc3 = layers.Concatenate()([u3, crop3])
    c8 = layers.Conv2D(128, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(conc3)
    c8 = layers.Conv2D(128, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(c8)
    c8 = layers.Dropout(dropout)(c8)
    u4 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c8) #crop4 = layers.Cropping2D(88)(c1)
    crop4 = c1
    conc4 = layers.Concatenate()([u4, crop4])
    c9 = layers.Conv2D(64, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(conc4)
    c9 = layers.Conv2D(64, 4, kernel_initializer='he_normal', activation=keras.layers.LeakyReLU(alpha=alfa), padding='same')(c9)
    c9 = layers.Dropout(finalDropout)(c9)
    c10 = layers.Conv2D(1, 1, activation=None, kernel_initializer='he_normal')(c9)
    model = keras.Model(inputs=[ip], outputs=[c10])
    return model


input_shape = (None, None,12)
unet = make_unet(input_shape) 
#unet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learnRate),
#        loss=weighted_mse)
unet.compile(optimizer=keras.optimizers.Adam(learning_rate=learnRate),
	loss=weighted_mse)


print('Loading data')

taiwanData = scipy.io.loadmat(inputFile)

#  Extract out the various variables so that we can work with them.  There
#  may be an easier way to do this automatically, but for now I'm just going 
# to go through them one by one.

rrs = taiwanData['rrs']

t6s = taiwanData['t6s']
t10s = taiwanData['t10s']

cbs = taiwanData['cbs']
cts = taiwanData['cts']

r1s = taiwanData['r1s']
r2s = taiwanData['r2s']
r3s = taiwanData['r3s']

t1s = taiwanData['t1s']
t2s = taiwanData['t2s']
t3s = taiwanData['t3s']

#  Here's where we load the other file.

print('Scaling data')

# We are going to have to first scale the data.  This means getting rid of
# NaNs and scaling from 0 to 1.

tmin = 190
tmax = 340

t10s = (t10s - tmin)/(tmax - tmin)
t6s = (t6s - tmin)/(tmax - tmin)

rrs[rrs<0]=0
rrs[rrs>150]=150

r1s=r1s/160
r2s=r2s/160
r3s=r3s/160

t1s=t1s/160
t2s=t2s/160
t3s=t3s/160

cbs[cbs<0]=0

cbs=cbs/20000
cts=cts/20000

rrs = np.log10(rrs+1)/2.2

print(np.shape(rrs))

#  Here is where we are going to add the terrain data. I've already acquired 
#  the elevation data, interpolated it to this grid, and normalized it to a log
#  scale. Therefore, no scaling needs to be done with it. There are two things
#  being added here: a water mask and the elevation data.

elevData = scipy.io.loadmat('/home/timw/input/elev_grid.mat')

ev = elevData['elevGrid']/3568
io = elevData['isOcean'] 

ev = np.repeat(ev[:, :, np.newaxis], 1534, axis=2)
io = np.repeat(io[:, :, np.newaxis], 1534, axis=2)

#inputs0 = np.stack([t10s, t6s,cbs, cts, r1s, r2s, r3s, t1s, t2s, t3s],3)

inputs0 = np.stack([t10s, t6s,cbs, cts, r1s, r2s, r3s, t1s, t2s, t3s,ev,io],3)


#  Make sure there are no NaNs hanging out, by setting them to zero.

inputs0 = np.nan_to_num(inputs0)
rrs = np.nan_to_num(rrs)

#  Reorder input dimensions so that they're in the same order that tensorflow 
#  expects.

inputs1 = np.transpose(inputs0,(2, 0, 1,3))
rrs = np.transpose(rrs,(2,0,1))

print('Sorting into testing and validation')

sizeLim = np.size(inputs1,0)

s1=round(trainSplit*sizeLim)
s2=round(valSplit*sizeLim)+s1

print('There are ' + str(s1) + ' scenes in the training set')

trainData = inputs1[:s1,:,:,:]
valData = inputs1[s1:s2,:,:,:]
testData = inputs1[s2:-1,:,:,:]

trainLabels = np.expand_dims(rrs[:s1,:,:],3)
valLabels = np.expand_dims(rrs[s1:s2,:,:],3)
testLabels = np.expand_dims(rrs[s2:-1,:,:],3)



callbacks = [
	keras.callbacks.EarlyStopping(
        	monitor='val_loss',
        	patience=stopPatience,
		restore_best_weights=True)
    ]

history = unet.fit(trainData, 
                    trainLabels,
                    epochs=numEpochs, 
                    batch_size=batchSize,
                    steps_per_epoch=numSteps,
                    shuffle=True,
                    validation_data=(valData, valLabels),
                    callbacks=callbacks),

print('Generating predictions of test data')


predictions = unet.predict(testData)
mdic1 = {"predictions":predictions,"test":testLabels,'val':valLabels,'b':b \
	,'c':c,'alpha':alfa,'learnRate':learnRate}

# New: change the outputs to actual rrs, not the transformed versions.

testLabels2 = np.power(10,testLabels*2.6)-1
predictions2 = np.power(10,predictions*2.6)-1

mt = np.mean(testLabels2)
st = np.std(testLabels2)

mp = np.mean(predictions2)
sp = np.std(predictions2)

print('Saving predictions')

fn = '/home/timw/mats/trained_' + str(numEpochs) + '_' + str(batchSize) + \
	'_' + str(learnRate) + '_' + str(b) + 'y' + \
        str(c) + '_tc_with_dem_' + str(alfa) + '.mat'

scipy.io.savemat(fn,mdic1)

copyfile(fn,'/home/timw/trained.mat')


print(' ')
print('Alpha = ' + str(alfa))
print('b = ' + str(b))
print('c = ' + str(c))
print('Test Prediction:')
print('        Mean: ' + str(mp))
print('         Std: ' + str(sp))
print('Test Truth:')
print('        Mean: ' + str(mt))
print('         Std: ' + str(st))

#  Save model

tf.keras.models.save_model(unet, 'test1',overwrite=True)

print('Done.')
