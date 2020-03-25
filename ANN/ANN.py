import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
import pickle
import gzip
import os
from urllib import request
import numpy as np

def check_labels(labels):
	new_labels = []
	for i in range(len(labels)):
		arr = [0] * 10
		arr[labels[i]] = 1

		new_labels.append(np.array(arr))

	return np.array(new_labels)

#load data_set
url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
if not os.path.isfile("mnist.pkl.gz"):
    request.urlretrieve(url, "mnist.pkl.gz")

data_set = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(data_set, encoding='latin1')
data_set.close()

#==============================================

#model
model = Sequential()

#input layer
model.add(Dense(200, input_dim=784, activation='sigmoid'))

model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_set[0], check_labels(train_set[1]), 1024, 20)

score = model.evaluate(test_set[0], check_labels(test_set[1]), verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])