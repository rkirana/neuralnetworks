import numpy as np
import pandas as pd
import os
import datetime as datetime
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import OneHotEncoder


epochs=100
batchsize=100
learning_rate = 0.1
momentum=0.9
numhid1=50
numhid2=200
init_wt=0.01
show_training_CE_after=100
show_validation_CE_after=1000
numwords = 3

def train (epochs=100, batchsize=100, learning_rate = 0.1, momentum=0.9, numhid1=50, numhid2=200, init_wt=0.01, show_training_CE_after=100, show_validation_CE_after=1000, numwords = 3):
	# epochs: number of epochs to run
	# output learned weights, biases and vocabulary
	[train_input, train_target, valid_input, valid_target, test_input, test_target, vocab] = load_data (N=batchsize)
	vocab_size = len (vocab)
	##########INITIALIZE THE WEIGHTS
	# initialize weights to hidden layer 1 - 250 * 50 matrix
	word_embedding_weights = init_wt * np.random.standard_normal ((vocab_size, numhid1))
	# initialize weights to hidden layer 2 - 50 * 200 matrix
	embed_to_hid_weights = init_wt * np.random.standard_normal ((numwords*numhid1, numhid2))
	# embed to hidden weights delta - 50 * 1 matrix
	hid_to_output_weights = init_wt * np.random.standard_normal ((numhid2, vocab_size))
	# hiden bias
	hid_bias_delta = np.zeros ((numhid2, 1))
	output_bias_delta = np.zeros ((vocab_size, 1))
	hid_bias = np.zeros ((numhid2, 1))
	output_bias = np.zeros ((vocab_size, 1))
	###########INITIALIZE THE DELTA AND GRADIENT MATRICES
	word_embedding_weights_delta = np.zeros ((vocab_size, numhid1))
	word_embedding_weights_gradient = np.zeros ((vocab_size, numhid1))
	embed_to_hid_weights_delta = np.zeros ((numwords * numhid1, numhid2))
	hid_to_output_weights_delta = np.zeros ((numhid2, vocab_size))
	bias_delta = np.zeros ((numhid2, 1))
	output_bias_delta = np.zeros ((vocab_size,1))
	enc = OneHotEncoder (sparse=False)
	enc.fit (train_target.T)
	expanded_train_target = enc.transform (train_target.T).T

	count = 0
	tiny = np.exp (-30)
	N=100
	# add batch information
	trainrows = train_input.shape [1]

	for epoch in np.arange (epochs):
		print ("Epoch %d", epoch)
		this_chunk_CE = 0
		trainset_CE = 0
		# Loop over mini-batches
		for m in arange (numbatches):
			input_batch = train_input [:,m*N:min((m+1)*N, trainrows)]
			target_batch = train_target [:,m*N:min((m+1)*N, trainrows)]
			input_batch = input_batch - 1
			target_batch = target_batch - 1
			###Forward layer
			[embedding_layer_state, hidden_layer_state, output_layer_state] = fprop (input_batch, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias, numhid1, numwords,  N,  vocab_size, vocab)


			# Forward propagate - calculate state of each layer
			embedding_layer_state = word_embedding_weights [input_batch.flatten()].reshape ( numhid1*numwords,-1)
			inputs_to_hidden_units =  np.dot ( embed_to_hid_weights.T, embedding_layer_state) + np.repeat (hid_bias, N, axis=1)
			hidden_layer_state = 1/(1+exp(-inputs_to_hidden_units))
			
			# inputs to softmax
			inputs_to_softmax = np.dot ( hid_to_output_weights.T, hidden_layer_state) + np.repeat (output_bias, N, axis=1)
			inputs_to_softmax = inputs_to_softmax - np.full ((  vocab_size, 1), np.max(inputs_to_softmax))

			# output layer state
			output_layer_state = exp (inputs_to_softmax)
			output_layer_state = output_layer_state/np.full (( vocab_size, 1), sum(output_layer_state))

			# Compute Derivative
		
			expanded_target_batch = enc.transform (target_batch.T).T

			error_deriv = output_layer_state - expanded_target_batch
			# Measure Loss function
			CE = -sum (expanded_target_batch * log (output_layer_state + tiny))/batchsize
			count = count + 1
			this_chunk_CE = this_chunk_CE + (CE-this_chunk_CE)/count
			trainset_CE = trainset_CE + (CE-trainset_CE)/(m+1)
			print ("Batch %d Train CE %.3f", m, this_chunk_CE)
			# Back Propagate - output layer
			hid_to_output_weights_gradient = np.dot ( hidden_layer_state, error_deriv.T )
			output_bias_gradient = error_deriv.sum (axis=1)
			output_bias_gradient = output_bias_gradient.reshape (vocab_size, -1)
			back_propagated_deriv_1 = np.dot ( hid_to_output_weights, error_deriv) *  hidden_layer_state * (1-hidden_layer_state)
			###Hidden Layer
			embed_to_hid_weights_gradient = np.dot ( embedding_layer_state, back_propagated_deriv_1.T)
			hid_bias_gradient = back_propagated_deriv_1.sum(axis=1)
			hid_bias_gradient = hid_bias_gradient.reshape (numhid2, -1)
			back_propagated_deriv_2 = np.dot( embed_to_hid_weights, back_propagated_deriv_1) 
			word_embedding_weights_gradient = np.zeros ((vocab_size, numhid2))
			## Embedding Layer
			for w in np.arange (numwords):
				word_embedding_weights_gradient = word_embedding_weights_gradient + np.dot(  enc.transform (input_batch[w].reshape(1,-1).T).T, back_propagated_deriv_2 [w *numhid1: (w+1)* numhid1,:].T)
			# update weights & biases
			word_embedding_weights_delta = momentum * word_embedding_weights_delta + word_embedding_weights_gradient / batchsize
			word_embedding_weights = word_embedding_weights - learning_rate * word_embedding_weights_delta
			embed_to_hid_weights_delta = momentum * embed_to_hid_weights_delta + embed_to_hid_weights_gradient/batchsize
			embed_to_hid_weights = embed_to_hid_weights - learning_rate * embed_to_hid_weights_delta
			hid_to_output_weights_delta = momentum * hid_to_output_weights_delta + hid_to_output_weights_gradient /batchsize
			hid_to_output_weights = hid_to_output_weights - learning_rate * hid_to_output_weights_delta
			hid_bias_delta = momentum * hid_bias_delta + hid_bias_gradient /batchsize
			hid_bias = hid_bias - learning_rate *hid_bias_delta
			output_bias_delta = momentum * output_bias_delta + output_bias_gradient/batchsize
			output_bias = output_bias - learning_rate * output_bias_delta
			#Validate
			[embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(valid_input, word_embedding_weights, embed_to_hid_weights,  hid_to_output_weights, hid_bias, output_bias)
			datasetsize = valid_input.shape [1]
			expanded_valid_target = expansioin_matrix (valid_target)
			CE = -sum (expanded_valid_target * log (output_layer_state + tiny) ) /datasetsize
			print ("Validation CE %.3f", CE)

# Evaluation on Validaiton set
[embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(valid_input, word_embedding_weights, embed_to_hid_weights,         hid_to_output_weights, hid_bias, output_bias)
datasetsize = valid_input.shape [1]
expanded_valid_target = expansion_matrix (valid_target)
CE = -sum (expanded_valid_target * log (output_layer_state + tiny) ) /datasetsize
print ("Final Validation CE %.3f", CE)


# Evaluation on Test set
[embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(test_input, word_embedding_weights, embed_to_hid_weights,        hid_to_output_weights, hid_bias, output_bias);
datasetsize = test_input.shape [1]
expanded_test_target = expansion_matrix(:, test_target);
CE = -sum(expanded_test_target .* log(output_layer_state + tiny))) / datasetsize;
print ("Final Test CE %.3f", CE)
model.word_embedding_weights = word_embedding_weights;
model.embed_to_hid_weights = embed_to_hid_weights;
model.hid_to_output_weights = hid_to_output_weights;
model.hid_bias = hid_bias;
model.output_bias = output_bias;
model.vocab = vocab;


























		






	



