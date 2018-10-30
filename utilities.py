import pandas as pd 
import numpy as np
import os
import random
from sklearn.metrics import confusion_matrix
import keras.backend as K
from itertools import product

class MyList(list):
    def __init__(self, *args):
        super(MyList, self).__init__(args)

    def __sub__(self, other):
        return self.__class__(*[item for item in self if item not in other])

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask
    
def weighted_categorical_crossentropy(y_true, y_pred, weights):
    
	weights = K.variable(weights)
        
	# scale predictions so that the class probas of each sample sum to 1
	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
	# clip to prevent NaN's and Inf's
	y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
	# calc
	loss = y_true * K.log(y_pred) * weights
	loss = -K.sum(loss, -1)
	return loss

def get_frail_percentage(model, pre_frail_inputs):
	pre_frail_predictions = model.predict(pre_frail_inputs)
	pre_frail_predictions = pre_frail_predictions.round()
	pre_frail_predictions = [ np.argmax(pred, axis = 0) for pred in pre_frail_predictions ]
	frail_prediction_percentage = float(pre_frail_predictions.count(1)) / float(len(pre_frail_predictions))
	
	return frail_prediction_percentage
	
def classify_patient(model, X_val, y_val, patients_for_val, patients_val):
	val_dict = {el:[0,0,0] for el in patients_for_val}
	
	for i in xrange(0, X_val.shape[0]):
		if y_val[i] == 0.0:
			val_dict[patients_val[i][0]][0] = 0
		else:
			val_dict[patients_val[i][0]][0] = 1
	
	#pred_val = model.predict([X_val, X_val])
	pred_val = model.predict(X_val)
	
	#for i in range(0, len(pred_val)):
	#	print val_dict[patients_val[i][0]][0], pred_val[i]
	#print(pred_val)
	pred_val = pred_val.round()
	
	pred_val = [ np.argmax(pred, axis = 0) for pred in pred_val]

	for i in range(0, len(pred_val)):
		if pred_val[i] == 0:
			val_dict[patients_val[i][0]][1] += 1
		else:
			val_dict[patients_val[i][0]][2] += 1
		
	for patient in val_dict:
		if val_dict[patient][1] > val_dict[patient][2]:
			print("Patient ID: %d, Label: %d, Prediction: %d"%(patient, val_dict[patient][0], 0))
			return [val_dict[patient][0], 0]
		else:
			print("Patient ID: %d, Label: %d, Prediction: %d"%(patient, val_dict[patient][0], 1))
			return [val_dict[patient][0], 1]


def read_data(data_path, split = "train"):
	""" Read data """

	# Fixed params
	n_class = 6
	n_steps = 128

	# Paths
	path_ = os.path.join(data_path, split)
	path_signals = os.path.join(path_, "Inertial_Signals")

	# Read labels and one-hot encode
	label_path = os.path.join(path_, "y_" + split + ".txt")
	labels = pd.read_csv(label_path, header = None)

	# Read time-series data
	channel_files = os.listdir(path_signals)
	channel_files.sort()
	n_channels = len(channel_files)
	posix = len(split) + 5

	# Initiate array
	list_of_channels = []
	X = np.zeros((len(labels), n_steps, n_channels))
	i_ch = 0
	for fil_ch in channel_files:
		channel_name = fil_ch[:-posix]
		dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
		X[:,:,i_ch] = dat_.as_matrix()

		# Record names
		list_of_channels.append(channel_name)

		# iterate
		i_ch += 1

	# Return 
	return X, labels[0].values, list_of_channels

def standardize(train, val, test):
	""" Standardize data """

	# Standardize train and test
	X_train = (train - np.mean(train, axis=0)[None,:,:])# / np.std(train, axis=0)[None,:,:]
	X_val = (val - np.mean(train, axis=0)[None,:,:])# / np.std(train, axis=0)[None,:,:]
	X_test = (test - np.mean(train, axis=0)[None,:,:])# / np.std(train, axis=0)[None,:,:]

	return X_train, X_val, X_test

def one_hot(labels, n_class = 6):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = []
	for i in labels:
		y.append(expansion[int(i)])
	y = np.asarray(y)

	return y

def get_batches(X, y, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]

def shuffle_randomly(a, b):
    
    a_shuf = []
    b_shuf = []
    
    index_shuf = range(len(a))
    random.shuffle(index_shuf)
    
    for i in index_shuf:
        a_shuf.append(a[i])
        b_shuf.append(b[i])
		
    return a_shuf, b_shuf
	
def shuffle_randomly3(a, b, c):
    
    a_shuf = []
    b_shuf = []
    c_shuf = []
    
    index_shuf = range(len(a))
    random.shuffle(index_shuf)
    
    for i in index_shuf:
        a_shuf.append(a[i])
        b_shuf.append(b[i])
        c_shuf.append(c[i])
		
    return a_shuf, b_shuf, c_shuf
    
def split_data(inputs, targets, patients, patients_for_val):
	
	X_train_size = X_test_size = X_val_size = 0
	
	patients_for_train = [item for item in list(np.unique(patients)) if item not in patients_for_val]
	idpatients_val = []
	for patient in patients_for_val:
		idpatient = [i for i, x in enumerate(patients) if x == patient]
		idpatients_val.append(idpatient)
	idpatients_val = [item for sublist in idpatients_val for item in sublist]
	X_val = inputs[idpatients_val]
	y_val = targets[idpatients_val]
	print(X_val.shape)
	
	#idpatients_not_train = idpatients_test + idpatients_val
	idpatients_train = list(set([i for i in range(inputs.shape[0])]) - set(idpatients_val))
	inputs = inputs[idpatients_train]
	targets = targets[idpatients_train]
	print(inputs.shape)
	
	print(np.unique(y_val))
	print(np.unique(targets))

	return inputs, targets, X_val, y_val, patients[idpatients_val], patients[idpatients_train]
	
def segmentize(inputs, labels, patients, window_size):
    new_inputs = inputs.reshape(int(inputs.shape[0] * inputs.shape[1] / window_size), window_size, inputs.shape[2])
    new_labels = []
    new_patients = []
    for label in labels:
        for i in range(int(inputs.shape[1] / window_size)):
            new_labels.append(label)
    for patient in patients:
        for i in range(int(inputs.shape[1] / window_size)):
            new_patients.append(patient)
    return new_inputs, new_labels, new_patients
