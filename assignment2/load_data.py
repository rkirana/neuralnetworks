
############################################################
###############LOAD DATA ##################################
#############################################################

import numpy as np
import pandas as pd
import os
import datetime as datetime
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def load_data (N=100):
	data = sp.io.loadmat('data.mat')
	train_data = data ['data']['trainData']
	valid_data = data ['data']['validData']
	test_data = data ['data']['testData']
	training = train_data[0][0].T
	validation = valid_data [0][0].T
	testing= test_data [0][0].T
	vocab = data ['data']['vocab'][0][0]
	myvocab = []
	for i in arange (0, 250):
		myvocab.append (vocab[0,i].tolist()[0])
	myvocab = map(lambda x: x.encode('ascii'), myvocab)
	D  = 3
	N=  100 # size of mini batch
	M = training.shape [0]/N
	train_input = training [:,:3].T
	train_target = training[:,3].reshape (1, -1)
	valid_input = validation [:,:3].T
	valid_target = validation [:,3].reshape (1, -1)
	test_input = testing [:,:3].T
	test_target = testing[:,3].reshape (1, -1)
	vocab_old = copy (vocab)
	vocab = copy (myvocab)
	train_target [0, 369000] = 24
	train_input = train_input - 1
	train_target = train_target - 1	
	valid_input = valid_input - 1
	valid_target = valid_target - 1
	test_input = test_input -1
	test_target = test_target - 1
	return ([train_input, train_target, valid_input, valid_target, test_input, test_target, vocab])


