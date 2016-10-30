
# Learns the weights of a perceptron and displays the results. # Learns the weights of a perceptron for a 2-dimensional dataset and plots
# the perceptron at each iteration where an iteration is defined as one full pass through the data. If a generously feasible weight vector% is provided then the visualization will also show the distance of the learned weight vectors to the generously feasible weight vector.
	# Parameters
		# neg_examples_nobias - The num_neg_examples x 2 matrix for the examples with target 0
		# pos_examples_nobias - The num_pos_examples x 2 matrix for the examples with target 1 
		# w_init - A 3-dimensional initial weight vector. The last element is the bias
		# w_gen_feas - A generously feasible weight vector	
	# Returns:   
		# w - The learned weight vector

def learn_perceptron (neg_examples_nobias, pos_examples_nobias, w_init, w_gen_feas):
	num_neg_examples = neg_examples_nobias.shape [0]
	num_pos_examples = pos_examples_nobias.shape [0]
	num_err_history = None
	w_dist_history = None

	neg_examples = np.concatenate ((neg_examples_nobias, np.ones ((num_neg_examples, 1))), axis=1)
	pos_examples = np.concatenate ((pos_examples_nobias, np.ones ((num_pos_examples, 1))), axis=1)


	if w_init is None:
		w_init = np.random.standard_normal ((1, 3))

	w = w_init
	iter = -1
	num_errs = 100
	num_err_history = []
	w_dist_history = []

	while num_errs > 0:
		iter = iter + 1
		# Find the data points that the perceptron has incorrectly classified and record the number of errors it makes
		[mistakes0, mistakes1] = eval_perceptron(neg_examples,pos_examples,w)
		num_errs = len (mistakes0) + len (mistakes1)
	
		num_err_history.append (num_errs)
		print ('Number of errors in iteration %d:%d',iter,num_errs)
		print ('weights:%s\n', w );
		#plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history);
		#key = input('<Press enter to continue, q to quit.>', 's');
		#if (key == 'q')
	   #	 return;
		#end
		#If a generously feasible weight vector exists, record the distance to it from the initial weight vector
		if w_gen_feas is not None:
			w_dist_history.append = np.linalg.norm (w - w_gen_feas)
   	#Update the weights of the perceptron.
		w = update_weights(neg_examples,pos_examples,w)



# Updates the weights of the perceptron for incorrectly classified points using the perceptron update algorithm. This function makes one sweep over the dataset.
# Inputs:  
#	neg_examples - The num_neg_examples x 3 matrix for the examples with target 0
#	num_neg_examples is the number of examples for the negative class.
#  pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
#  num_pos_examples is the number of examples for the positive class.
#   w_current - A 3-dimensional weight vector, the last element is the bias.
# Returns:
#   w - The weight vector after one pass through the dataset using the perceptron
#  learning rule.


def update_weights (neg_examples, pos_examples, w_current, learning_rate = 1):
	w = w_current;
	num_neg_examples = neg_examples_nobias.shape [0]
	num_pos_examples = pos_examples_nobias.shape [0]

	for i in xrange (num_neg_examples):
		 x = neg_examples [i,:].reshape (1, neg_examples_nobias.shape[1]+1)
		 activation = np.dot (x, w.T)
		 if (activation >= 0):
		 	deltaw = learning_rate * x * -1
			w = w + deltaw
		
	for i in xrange (num_pos_examples):
		 x = pos_examples [i,:].reshape (1, pos_examples_nobias.shape[1]+1)
		 activation = np.dot (x, w.T)
		 if (activation >= 0):
		 	deltaw = learning_rate * x * -1
			w = w + deltaw

	return w

def eval_perceptron (neg_examples, pos_examples, w):
# Evaluates the perceptron using a given weight vector. Here, evaluation refers to finding the data points that the perceptron incorrectly classifies
# Inputs:
#   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
#   num_neg_examples is the number of examples for the negative class.
#   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
#   num_pos_examples is the number of examples for the positive class.
#   w - A 3-dimensional weight vector, the last element is the bias.
# Returns:
#   mistakes0 - A vector containing the indices of the negative examples that have been incorrectly classified as positive.
#  mistakes0 - A vector containing the indices of the positive examples that have been incorrectly classified as negative.
	num_neg_examples = neg_examples_nobias.shape [0]
	num_pos_examples = pos_examples_nobias.shape [0]
	mistakes0 = []
	mistakes1 = []
	for i in xrange (num_neg_examples):
		 x = neg_examples [i,:].reshape (1, neg_examples_nobias.shape[1]+1)
		 activation = np.dot (x, w.T)
		 if (activation >= 0):
		     mistakes0.append (i)
	for i in xrange (num_pos_examples):
		 x = pos_examples [i,:].reshape (1, pos_examples_nobias.shape[1]+1)
		 activation = np.dot (x, w.T)
		 if (activation >= 0):
		     mistakes1.append (i)
	return ([mistakes0, mistakes1])


