
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

data = sp.io.loadmat('data.mat')
train_data = data ['data']['trainData']
valid_data = data ['data']['validData']
test_data = data ['data']['testData']

training = train_data[0][0].T
validation = valid_data [0][0].T
testing= test_data [0][0].T
vocab = data ['data']['vocab'][0][0]

myvocab = ['dummy']
for i in arange (0, 250):
	myvocab.append (vocab[0,i].tolist()[0])

myvocab = map(lambda x: x.encode('ascii'), myvocab)

D  = 3
N=  100 # size of mini batch
M = training.shape [0]/N

train_input = training [:,:3]
train_target = training[:,3]

valid_input = validation [:,:3]
valid_target = validation [:,3]

test_input = testing [:,:3]
test_target = testing[:,3]

vocab_old = copy (vocab)
vocab = copy (myvocab)


############################################################
###############WORD DISTANCE##################################
#############################################################
def word_distance (word1, word2, vocab):
	if word1 not in vocab:
		print ("Word %s not in vocabulary\n", word1)
	if word2 not in vocab:
		print ("Word %s not in vocabulary\n", word2)
		
		
		
	
	
	
############################################################
###############DISPLAY NEAREST WORDS#####################
#############################################################
def display_nearest_words (word, model, k)







	
############################################################
###############TRAIN NEURAL NETWORK MODEL#####################
#############################################################


