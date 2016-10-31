import numpy as np
import pandas as pd
import os
import datetime as datetime
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt


%cd '/home/kirana/coursera/neuralnetworks/Assignment1/Datasets'



data1 = sp.io.loadmat('dataset1.mat')
data2 = sp.io.loadmat('dataset2.mat')
data3 = sp.io.loadmat('dataset3.mat')
data4 = sp.io.loadmat('dataset4.mat')

neg_examples_nobias = data4['neg_examples_nobias']
pos_examples_nobias = data4['pos_examples_nobias']
w_init =data4['w_init'].T
w_gen_feas = data4['w_gen_feas'].T


learn_perceptron (data1['neg_examples_nobias'], data1 ['pos_examples_nobias'], data1['w_init'], data1['w_gen_feas'])

learn_perceptron (data2['neg_examples_nobias'], data2 ['pos_examples_nobias'], data2['w_init'], data2['w_gen_feas'])

learn_perceptron (data3['neg_examples_nobias'], data3 ['pos_examples_nobias'], data3['w_init'], data3['w_gen_feas'])


learn_perceptron (data4['neg_examples_nobias'], data4 ['pos_examples_nobias'], data4['w_init'], data4['w_gen_feas'])


