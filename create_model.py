import numpy as np
import seaborn as sns
import utils
import os
import prepare_data as pd

from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image
import tensorflow as tf
import time

dense_layers = [2]
layer_sizes = [256, 512]
conv_layers = [3, 4]

for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:

			NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
			tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
			print(NAME)

			#Initializing CNN
			model = Sequential()
			model.add(Conv2D(64, (3,3), padding='same', input_shape=(48,48,1)))
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))
			model.add(Dropout(0.25))

			for l in range(conv_layer-1):

				#Convolution layer
				model.add(Conv2D(layer_size, (3,3), padding='same'))
				model.add(BatchNormalization())
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))
				model.add(Dropout(0.25))

			# Flattening
			model.add(Flatten())

			for l in range(dense_layer):

				# Fully connected layer
				model.add(Dense(layer_size))
				model.add(BatchNormalization())
				model.add(Activation('relu'))
				model.add(Dropout(0.25))

			model.add(Dense(7, activation='softmax'))

			opt = Adam(lr=0.0005)
			model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
			model.summary()

			epochs = 15
			steps_per_epoch = pd.train_generator.n//pd.train_generator.batch_size
			validation_steps = pd.validation_generator.n//pd.validation_generator.batch_size

			reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, mode='auto')
			checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
			callbacks = [checkpoint, reduce_lr, tensorboard]


			history = model.fit(
			    x=pd.train_generator,
			    steps_per_epoch=steps_per_epoch,
			    epochs=epochs,
			    validation_data = pd.validation_generator,
			    validation_steps = validation_steps,
			    callbacks=callbacks
			)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)