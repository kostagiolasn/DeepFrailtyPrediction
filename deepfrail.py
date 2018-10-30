import numpy as np
import os
from utilities import *
import matplotlib.pyplot as plt
import h5py
import random
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import Flatten, Activation, BatchNormalization, Merge
from keras.layers import Reshape, Dropout
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
from custom_metrics import BalancedAccuracy
from sklearn.cross_validation import KFold
from functools import partial
from functools import partial
import matplotlib.cm as cm
import pylab as pl
import csv
import tensorflow as tf
from keras import backend as K


random.seed(1)
np.random.seed(1337)
# load data from CSV file to tensorflow shared memory variables
h5f = h5py.File('strap_constantFrailty.mat','r')
inputs = h5f['X'][:]
labels = h5f['Y']
patients = h5f['patientID']

inputs = np.einsum('ijk->kji', inputs)
labels = np.einsum('ij->ji', labels)
patients = np.einsum('ij->ji', patients)

# Unomment the following for multi-class classification
"""
index1 = 0
for i in range(0, labels.shape[0]):
	if labels[i][0] == 1.0:
		break
	index1 += 1
	
index2 = 0
for i in range(0, labels.shape[0]):
	if labels[i][0] == 2.0:
		break
	index2 += 1
	
print(index1, index2)
"""
inputs = np.asarray(list(inputs[:14210]) + list(inputs[31744:]))
labels = np.asarray(list(labels[:14210]) + list(labels[31744:]))
patients = np.asarray(list(patients[:14210]) + list(patients[31744:]))

unique_patients = np.unique(patients)

pre_frail_inputs = np.asarray(list(inputs[14211:31744]))
pre_frail_labels = np.asarray(list(labels[14211:31744]))
pre_frail_patients = np.asarray(list(patients[14211:31744]))

# Comment the following for multi-class classification
for i in range(len(labels)):
	if labels[i] == 2.0:
		labels[i] = 1.0

# Uncomment the following to shuffle the data randomly
"""
inputs, labels, patients = shuffle_randomly3(inputs, labels, patients)

for i in labels:
	print i

inputs = np.asarray(inputs)
labels = np.asarray(labels)
patients = np.asarray(patients)
"""

print(inputs.shape, labels.shape, patients.shape)

# remove two patients to facilitate kfold split
unique_patients = list(unique_patients)
unique_patients.remove(1006)
unique_patients.remove(1007)
unique_patients = np.asarray(unique_patients)
np.random.shuffle(unique_patients)
print(unique_patients)

kf = KFold(unique_patients.shape[0], 5)
count = 0
results = []
for train_index, val_index in kf:
	
	#if count > 22:
	patients_for_val = unique_patients[val_index]	
	X_train, y_train, X_val, y_val, patients_val, patients_train = split_data(inputs, labels, patients, patients_for_val)
	
	X_train, y_train, patients_train = shuffle_randomly3(X_train, y_train, patients_train)
	X_val, y_val, patients_val = shuffle_randomly3(X_val, y_val, patients_val)
		
	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
	X_val = np.asarray(X_val)
	y_val = np.asarray(y_val)
	patients_val = np.asarray(patients_val)

	X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis = 0)
	X_val = (X_val - np.mean(X_train, axis=0)) / np.std(X_train, axis = 0)
	
	if count == 0:
		pre_frail_inputs = (pre_frail_inputs - np.mean(X_train, axis=0)) / np.std(X_train, axis = 0)
		pre_frail_inputs = np.reshape(pre_frail_inputs, (pre_frail_inputs.shape[0], pre_frail_inputs.shape[1], pre_frail_inputs.shape[2], 1))
	
	"""
	# Uncomment here for weighted loss function
	w_array = np.ones((2,2))
	w_array[0,0] = 1
	w_array[0,1] = float(len(y_train)) / float(list(y_train).count(0))
	w_array[1,0] = float(len(y_train)) / float(list(y_train).count(1))
	w_array[1,1] = 1
	print(w_array)
	
	ncce = partial(w_categorical_crossentropy, weights = w_array)
	ncce.__name__ = 'w_categorical_crossentropy'
	"""
	
	# Weighted loss function to tackle class imbalance in the dataset
	weights = np.array([float(len(y_train)) / float(list(y_train).count(0)), float(len(y_train)) / float(list(y_train).count(1))])
	loss = partial(weighted_categorical_crossentropy, weights = weights)
	loss.__name__ = 'weighted_categorical_crossentropy'

	#print(X_train.shape, X_val.shape)

	y_train_one_hot = one_hot(y_train, 2)
	y_val_one_hot = one_hot(y_val, 2)

	print("Training set: %d, Validation set: %d"%(y_train.shape[0], y_val.shape[0]))

	print(X_train.shape)

	# Set a few callbacks to reduce the learning rate once the model starts to level off
	reduce_lr1=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', \
		factor=0.5, patience=12, verbose=1, mode='auto', epsilon=0.01, cooldown=1000, min_lr=0)
	reduce_lr2=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', \
		factor=0.5, patience=12, verbose=1, mode='auto', epsilon=0.005, cooldown=1000, min_lr=0)
	reduce_lr3=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', \
		factor=0.5, patience=12, verbose=1, mode='auto', epsilon=0.001, cooldown=1000, min_lr=0)

	# Set balanced accuracy callback
	balanced_accuracy = BalancedAccuracy(train_data = (X_train, y_train), validation_data = (X_val, y_val))

	#stop training before model starts to overfit
	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', \
		min_delta=0., patience=15, verbose=1, mode='auto')
	
	# Uncomment accordingly
	#CALLBACKS = [early_stop,reduce_lr1,reduce_lr2,reduce_lr3]#+[model_checkpoint]
	#CALLBACKS = [reduce_lr1,reduce_lr2,reduce_lr3,balanced_accuracy]
	CALLBACKS = [balanced_accuracy]
	
	
	model = Sequential()
	#model.add(Reshape((X_train.shape[1], X_train.shape[2]), input_shape=(1, X_train.shape[1], X_train.shape[2])))
	model.add(Conv1D(32, 3, padding='same', input_shape=(X_train.shape[1], X_train.shape[2])))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	print(model.output_shape)
	#model.add(Flatten())
	model.add(GlobalAveragePooling1D())
	#model.add(Dense(256, init='uniform'))
	#model.add(BatchNormalization())
	#model.add(Activation('relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(128, kernel_initializer="uniform"))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation = 'softmax'))
	

	"""
	# Uncomment to use both LSTM and CNN extracted features, 
	# just remember to change X_val to [X_val, X_val] as the 
	# architecture takes 2 tensor matrices as inputs.
	
	right = Sequential()
	right.add(Reshape((X_train.shape[1], X_train.shape[2]), input_shape=(X_train.shape[1], X_train.shape[2], 1)))
	right.add(Convolution1D(32, 3, activation='relu'))#, input_shape=(X_train.shape[1], X_train.shape[2])))
	right.add(BatchNormalization())
	right.add(Dropout(0.5))
	right.add(Flatten())
	right.add(Dense(256, init='uniform'))
	right.add(BatchNormalization())
	right.add(Activation('relu'))
	right.add(Dropout(0.5))
	right.add(Dense(128, init='uniform'))
	right.add(BatchNormalization())
	right.add(Activation('relu'))
	right.add(Dropout(0.5))

	# design network
	left = Sequential()
	left.add(Reshape((X_train.shape[1], X_train.shape[2]), input_shape=(X_train.shape[1], X_train.shape[2], 1)))
	left.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))

	model = Sequential()
	model.add(Merge([left, right], mode='concat'))
	model.add(Dense(128, init='uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation = 'softmax'))
	"""

	# Uncomment here if you want to save your model
	#model.load_weights("weights-improvement-73-0.78.hdf5")
	#adam = Adam(lr = 0.000001)
	sgd = SGD(lr=0.0001)
	model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
	# fit network
	history = model.fit(X_train, y_train_one_hot, epochs=30, batch_size=128, validation_data=(X_val, y_val_one_hot), verbose=2, shuffle=True, callbacks=CALLBACKS)
	
	# Uncomment here if you want to save your model
	#model.save_weights("weights2.h5", overwrite = True)

	
	"""
	#Uncomment to use a data generator to augment the training set each epoch
	train_datagen = ImageDataGenerator(featurewise_center=False,
		samplewise_center=False,
		featurewise_std_normalization=False,
		samplewise_std_normalization=False,
		zca_whitening=False,
		#rotation_range=5.,
		width_shift_range=.2)
		#height_shift_range=.05),
		#shear_range=20*(3.1416/180),
		#zoom_range=0.05)
	

	train_datagen.fit(X_train)

	Batch_Size = 512
	history=my_model.fit_generator(train_datagen.flow(X_train,y_train_one_hot, \
				   batch_size=Batch_Size,shuffle=True), samples_per_epoch = X_train.shape[0], nb_epoch=42,\
				   verbose=True, validation_data=(X_val,y_val_one_hot))
	"""
	
	#print(model.layers[1].get_weights()[0].shape)
	"""
	# Uncomment to visualize learned weights of CNN
	weight_conv1d = model.layers[1].get_weights()[0][:,:,0,:]
	print(weight_conv1d.shape)
	col_size = 4
	row_size = 8
	filter_index = 0
	fig, ax = plt.subplots(row_size, col_size, figsize=(8, 20))
	for row in range(0, row_size):
		for col in range(0, col_size):
			ax[row][col].imshow(weight_conv1d[:,:,filter_index], cmap = "gray", aspect = 'auto')
			filter_index += 1
			
	plt.show()
	"""
	
	y_val_pred = model.predict(X_val)
	patient_dict = dict.fromkeys(np.unique(patients_val))
	
	for i in patient_dict:
		patient_dict[i] = [0, 0, 0]
	
	for i in range(y_val_pred.shape[0]):
		patient_dict[patients_val[i][0]][0] += 1
		patient_dict[patients_val[i][0]][1] += float(y_val_pred[i][0])
	for i in patient_dict:
		patient_dict[i][1] = float(patient_dict[i][1]) / float(patient_dict[i][0])
	
	for i in range(y_val_pred.shape[0]):
		 patient_dict[patients_val[i][0]][2] += (float(y_val_pred[i][0]) - patient_dict[patients_val[i][0]][1])**2
	
	
	for i in patient_dict:
		patient_dict[i][2] = float(patient_dict[i][2]) / float(patient_dict[i][0])
	"""
	with open('Pred-Mean_Pred-Std.Dev-Label-PatientID.csv', 'wb') as myfile:
		for i in range(y_val_pred.shape[0]):
			mylist = [y_val_pred[i][0], patient_dict[patients_val[i][0]][1], patient_dict[patients_val[i][0]][2], y_val[i][0], patients_val[i][0]]
			wr = csv.writer(myfile)
			wr.writerow(mylist)
	"""
	#for i in range(y_train_pred.shape[0]):
	#	f = open("Preds-Label-PatientID.txt","a+")
	#	f.write("%f %d %d\n" % (y_train_pred[i][0], y_train[i], patients_train[i]))
	#	f.close()
	
	"""			   
	train_loss_history = history.history["loss"]
	numpy_train_loss_history = np.array(train_loss_history)
	title = "5fold_train_loss_history" + str(count)
	np.savetxt(title + ".txt", numpy_train_loss_history, delimiter=",")
	"""
	"""
	train_acc_history = history.history["acc"]
	numpy_train_acc_history = np.array(train_acc_history)
	title = "5fold_train_acc_history" + str(count)
	np.savetxt(title + ".txt", numpy_train_acc_history, delimiter=",")
	"""
	# plot history
	"""
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.plot(history.history['loss'], label='train loss')
	plt.plot(history.history['val_loss'], label='validation loss')
	plt.title('Loss per epoch, lr = 0.02')
	plt.legend()
	plt.show()

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.plot(history.history['acc'], label='train accuracy')
	plt.plot(history.history['val_acc'], label='validation accuracy')
	plt.title('Accuracy per epoch, lr = 0.02')
	plt.legend()
	plt.show()
	
	# Uncomment this line to get results on pre-frail patients
	#print("Percentage of pre-frail patients predicted as being frail: %.3f%%"%(get_frail_percentage(model, pre_frail_inputs) * 100))

	results.append(classify_patient(model, X_val, y_val, patients_for_val, patients_val))
	
	count += 1
	
print("Presenting the final predictions per patient")	
print(results)"""
