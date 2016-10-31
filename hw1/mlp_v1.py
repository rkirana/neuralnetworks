
import numpy as np
import numpy as np
import pandas as pd
import os
import datetime as datetime
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt


%cd '/home/kirana/coursera/neuralnetworks/Assignment1/Datasets'
data1 = sp.io.loadmat('dataset1.mat')

bigdata = np.concatenate ((data1['neg_examples_nobias'], data1['pos_examples_nobias']))
target = np.concatenate ((np.tile (0, data1['neg_examples_nobias'].shape[0]),np.tile (1, data1['pos_examples_nobias'].shape[0])))
target = target.reshape (bigdata.shape[0], 1)
bigdata = np.concatenate ((bigdata, target), axis=1)

inputarray = bigdata [:, :2]
outputarray = bigdata [:,2]
outputarray = outputarray.reshape (bigdata.shape[0], 1)
numneurons = 1
learningrate = 0.1
maxiterations = 10
bias = 0;
	inputarray = np.concatenate ((inputarray, np.ones ((inputarray.shape[0], 1))), axis=1)

perceptronlearningalgorithm (inputarray, outputarray)

def perceptronlearningalgorithm (inputarray, outputarray, learningrate = 1, maxiterations = 100, bias=1):
	# define the weight matrix
	#weightarray = np.zeros ((inputarray.shape[1], numneurons))
	weightarray = np.random.standard_normal ((inputarray.shape[1], 1))
	for i in xrange (maxiterations):
		z = np.dot (inputarray, weightarray) + bias
		activations = np.where (z > 0, 1, 0)
		errs = np.abs (outputarray - activations)
		weightarray += learningrate * np.dot (np.transpose (inputarray), ( outputarray - activations ))
		num_errs = np.sum (errs)
		if num_errs == 0:
			break
		print ("Iter %d: Number of Errors %d", i, num_errs)
		#print weightarray
	print activations
	return weightarray


def activation (outputs, type = 'linear'):
	if type == 'linear':
		return outputs
	if type == 'logistic':
		return 1.0/(1.0 + exp (-outputs)
	if type == 'softmax':
		normalizers = sum (exp(outputs), axis=1)*ones ((1,shape(outputs)[0]))
		return (exp(outputs)/normalizers)
	if type == 'tanh':
		return
	return outputs

def mlp (inputarray, outputarray, numneurons = 1, learningrate = 1, maxiterations = 100, bias=1, momentum = 1, flag_scale = 0, validinputarray = None):
	# number of input layer nodes
	numinputnodes = inputarray.shape[1]
	# define the weight matrix ij
	weightarrayij = np.random.standard_normal ((inputarray.shape[1], numneurons))
	weightarrayij = weightarrayij/sqrt (numinputnodes)
	# define the weight matrix jk
	weightarrayjk = np.random.standard_normal ((numneurons, outputarray.shape [1]))
	weightarrayjk = weightarrayjk/sqrt (numneurons)
	# hidden activations
	hidden = np.zeros (numneurons)

	# scale inputs
	if flag_scale == 1:
		inputarray = (inputarray - inputarray.mean (axis= 0))/inputarray.var (axis=0)


	# early stopping

	# Training
	for i in xrange (maxiterations):
		# Forwards phase
			# activation of each neuron in hidden layer
		hidden = activation (np.dot (inputarray, weightarrayij))
			# activations of the output layer
		hidden = hidden.T
		outputys = activation (np.dot (hidden, weightarrayjk))
		# Backwards Phase
			# error at outputs
		deltaoutput = (outputarray - outputys) * outputys * (1-outputys)
		deltahidden = hidden * (1-hidden) * np.dot (weightarrayjk, outputys)

		deltahidden = deltahidden * momentum
		deltaoutput = deltaoutput * momentum
		# Backpropagation
			# output to hidden
			weightarrayjk = weightarrayjk + learningrate * deltaoutput * hidden
			# hidden to input 
			weightarrayij = weightarrayij + learningrate * deltahidden * inputarray

		



		z = np.dot (inputarray, weightarray) + bias
		activations = np.where (z > 0, 1, 0)
		errs = np.abs (outputarray - activations)
		weightarray += learningrate * np.dot (np.transpose (inputarray), ( outputarray - activations ))
		num_errs = np.sum (errs)
		if num_errs == 0:
			break
		print ("Iter %d: Number of Errors %d", i, num_errs)
		#print weightarray
	print activations
	return weightarray






