
def fprop (input_batch, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias, numhid1, numwords,  vocab_size, vocab):
		inputshape = input_batch.shape [1]
		# Forward propagate - calculate state of each layer
		myarr = input_batch.flatten(order='F')
		myarr = myarr.tolist ()
		embedding_layer_state = word_embedding_weights [myarr].reshape ( (-1, numhid1*numwords)).T
		inputs_to_hidden_units =  np.dot ( embed_to_hid_weights.T, embedding_layer_state) + np.repeat (hid_bias, inputshape, axis=1)
		#hidden_layer_state = np.zeros ((numhid2, batchsize))
		#temp = -inputs_to_hidden_units
		#temp = 1+exp (temp)
		#temp = 1/temp
		#hidden_layer_state = temp
		hidden_layer_state = 1/(1+exp(-inputs_to_hidden_units))
		# inputs to softmax#inputs_to_softmax = np.zeros((vocab_size, inputshape))
		inputs_to_softmax = np.dot ( hid_to_output_weights.T, hidden_layer_state) + np.repeat (output_bias, inputshape, axis=1)
		
		inputs_to_softmax = inputs_to_softmax - np.repeat ( inputs_to_softmax.max (0).reshape (1,-1), vocab_size,0)
		# output layer state
		output_layer_state = exp (inputs_to_softmax)
		output_layer_state = output_layer_state/np.repeat (output_layer_state.sum (0). reshape (1,-1), vocab_size, 0)
		# return
		return ([embedding_layer_state, hidden_layer_state, output_layer_state])		


