'''''''''''''''''''''''''''''''''''''''''''''''
'			Import libs						  '
'''''''''''''''''''''''''''''''''''''''''''''''

#TEST IF libs exists
NUMPY_EXIST = False;
CV_EXIST = False;
MATPLOTLIB_EXIST = False;
TENSORFLOW_EXIST = False;
try:
	import cv2
	CV_EXIST = True;
except:
	print ("CV2 does not exist ")

try:
	import numpy as np
	NUMPY_EXIST = True
except:
	print ("NUMPY does not exist ")

try: 
	import matplotlib.pyplot as plt 
	MATPLOTLIB_EXIST = True
except:
	print ("MATPLOTLIB does not exist ")

try:
	import tensorflow as tf
	TENSORFLOW_EXIST = True
	from tf_utils import convert_to_one_hot
	import tf_utils
	from tf_wrappers import model
except:
	print ("tensorflow does not exist")

import sys


'''''''''''''''''''''''''''''''''''''''''''''''
'			Load you training data 			  '
'''''''''''''''''''''''''''''''''''''''''''''''
def load_dataset_your_own():

    train_set_x_orig=[]
    train_set_y_orig=[] 
    test_set_x_orig= []
    test_set_y_orig=[] 
    classes=[]
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def main():
	'''''''''''''''''''''''''''''''''''''''''''''''
	'			Load data 						  '
	'''''''''''''''''''''''''''''''''''''''''''''''
	#All available data sets
	data_sets=['sign_language','mnist']

	data_use = 'mnist'
	if(data_use == 'sign_language'):
		X_train, Y_train, X_test, Y_test, classes = tf_utils.load_dataset_sign_language()
	elif(data_use == 'mnist'):
		X_train, Y_train, X_test, Y_test, classes = tf_utils.load_dataset_mnist()





	'''''''''''''''''''''''''''''''''''''''''''''''
	'			Pre-process data 				  '
	'''''''''''''''''''''''''''''''''''''''''''''''
	


	print ("number of training examples = " + str(X_train.shape[1]))
	print ("number of test examples = " + str(X_test.shape[1]))
	print ("X_train shape: " + str(X_train.shape))
	print ("Y_train shape: " + str(Y_train.shape))
	print ("X_test shape: " + str(X_test.shape))
	print ("Y_test shape: " + str(Y_test.shape))



	print (X_train.shape[0])
	print (Y_train.shape[1])

	'''''''''''''''''''''''''''''''''''''''''''''''
	'			Train the model					  '
	'''''''''''''''''''''''''''''''''''''''''''''''

	#Specify number of layers

	layer_array= [X_train.shape[0],1000,1000,Y_train.shape[0]]

	#Train the model
	learning_rate = 0.0001
	num_epoch = 1000
	minibatch_size = 32

	parameters,costs = model(
		X_train, 
		Y_train, 
		X_test, 
		Y_test,
		layer_array, 
		learning_rate,    #learing rate
		num_epoch, #num epoch
		minibatch_size , #	minibatch_size 
		True #print cost
		)

	#write the parameters to file
	np.save('parameters.npy',parameters)

	'''''''''''''''''''''''''''''''''''''''''''''''
	'			Plot the  cost					  '
	'''''''''''''''''''''''''''''''''''''''''''''''

	if(MATPLOTLIB_EXIST):
		#plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()


if __name__ == "__main__":
  main()