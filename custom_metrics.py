from keras.callbacks import Callback
import numpy as np
import operator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from utilities import *

class BalancedAccuracy(Callback):
	def __init__(self, train_data, validation_data):
		super(BalancedAccuracy, self).__init__()
		self.acas = []
		self.validation_data = validation_data
		self.train_data = train_data
        
	def on_epoch_end(self, epoch, logs={}):

		X_val = self.validation_data[0]
		y_val = self.validation_data[1]
		
		X_train = self.train_data[0]
		y_train = self.train_data[1]
		
		y_val_pred = self.model.predict(X_val)		
		y_train_pred = self.model.predict(X_train)
		
		val_score = self.eval_avg_class_acc(y_val, y_val_pred)
		train_score = self.eval_avg_class_acc(y_train, y_train_pred)
		
		self.acas.append([val_score])
		self.acas.append([train_score])
		
		"""
		f = open("5fold_balanced_acc_val.txt","a+")
		f.write("%f\n" % (val_score))
		f.close()
		
		f = open("5fold_balanced_acc_train.txt","a+")
		f.write("%f\n" % (train_score))
		f.close()
		"""
		print "\nBalanced Accuracy - train: %.3f \t val: %.3f"%(train_score, val_score)        
        
	def eval_avg_class_acc(self, y_true, y_pred):
		
		# decode one-hot to single labels
		y_pred = y_pred.round()
		y_pred = [ np.argmax(pred, axis = 0) for pred in y_pred ]
		y_true = [ np.argmax(label, axis = 0) for label in y_true ]
		
		cf = confusion_matrix(y_true, y_pred)
		if np.unique(y_true).shape[0] == 2:
			sensitivity = float(cf[1][1]) / float((cf[1][1] + cf[1][0]))
			specificity = float(cf[0][0]) / float((cf[0][1] + cf[0][0]))

			balanced_acc = (sensitivity + specificity) / 2
		else:
			balanced_acc = accuracy_score(y_true, y_pred)
		
		return balanced_acc
