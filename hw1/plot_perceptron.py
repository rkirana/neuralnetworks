
import matplotlib.pyplot as plt

# Plots information about a perceptron classifier on a 2-dimensional dataset.
def  plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history):
# The top-left plot shows the dataset and the classification boundary given by the weights of the perceptron. The negative examples are shown as circles while the positive examples are shown as squares. If an example is colored  green then it means that the example has been correctly classified by the provided weights. If it is colored red then it has been incorrectly classified.The top-right plot shows the number of mistakes the perceptron algorithm has made in each iteration so far. The bottom-left plot shows the distance to some generously feasible weight vector if one has been provided (note, there can be an infinite number of these). Points that the classifier has made a mistake on are shown in red, while points that are correctly classified are shown in green.The goal is for all of the points to be green (if it is possible to do so).
# Inputs:
#   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0. 
#   num_neg_examples is the number of examples for the negative class.
#   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
#   num_pos_examples is the number of examples for the positive class.
#   mistakes0 - A vector containing the indices of the datapoints from class 0 incorrectly classified by the perceptron. This is a subset of neg_examples.
#   mistakes1 - A vector containing the indices of the datapoints from class 1 incorrectly classified by the perceptron. This is a subset of pos_examples.
#   num_err_history - A vector containing the number of mistakes for each iteration of learning so far.
#   w - A 3-dimensional vector corresponding to the current weights of the perceptron. The last element is the bias.
#   w_dist_history - A vector containing the L2-distance to a generously feasible weight vector for each iteration of learning so far.
#       Empty if one has not been provided.
	num_neg_examples = neg_examples.shape [0]
	num_pos_examples = pos_examples.shape [0]
	neg_correct_ind = [x for x in xrange (num_neg_examples) if x not in mistakes0]
	pos_correct_ind = [x for x in xrange (num_pos_examples) if x not in mistakes1]
	neg_incorrect_ind = [x for x in xrange (num_neg_examples) if x in mistakes0]
	pos_incorrect_ind = [x for x in xrange (num_pos_examples) if x in mistakes1]
	plt.subplot (2, 2, 1)
	if len (neg_correct_ind) > 0:
		plt.plot(neg_examples[neg_correct_ind,0],neg_examples[neg_correct_ind,1],'bo')
	if len(pos_correct_ind) > 0:
		plt.plot(pos_examples[pos_correct_ind,0],pos_examples[pos_correct_ind,1],'b+')
	if len (neg_incorrect_ind) > 0:
		plt.plot(neg_examples[neg_incorrect_ind,0],neg_examples[neg_incorrect_ind,1],'ro')
	if len(pos_incorrect_ind) > 0:
		plt.plot(pos_examples[pos_incorrect_ind,0],pos_examples[pos_incorrect_ind,1],'r+')
	plt.title('Classifier')
	plt.plot([-5,5],[(-w[0,2]+5*w[0,0])/w[0,1],(-w[0,2]-5*w[0,0])/w[0,1]])
	plt.xlim(-1,1)
	plt.ylim(-1,1)
	plt.subplot(2,2,2)
	plt.plot(arange (len(num_err_history)),num_err_history)
	plt.xlim(-1,max(15,len(num_err_history)))
	plt.ylim(0,num_neg_examples+num_pos_examples)
	plt.title('Number of errors')
	plt.xlabel('Iteration')
	plt.ylabel('Number of errors')
	plt.subplot(2,2,3)
	plt.plot(arange (len(w_dist_history)),w_dist_history)
	plt.xlim(-1,max(15,len(num_err_history)))
	plt.ylim(0,15)
	plt.title('Distance')
	plt.xlabel('Iteration')
	plt.ylabel('Distance')
	return


