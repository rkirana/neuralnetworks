def fprop (input_batch, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias):
	numwords = input_batch.shape [0]
	batchsize = input_batch.shape [1]
	numhid2 = embed_to_hid_weights.shape [1]
	# Compute state of word embedding layer
	embedding_layer_state = word_embedding_weights (input_batch.shape [1], 1 )
