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
	
except:
	print ("tensorflow does not exist")

import sys
if(TENSORFLOW_EXIST == True):
	from tf_utils import convert_to_one_hot
	import tf_utils
	import tf_CNN_wrapper


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
	print ("Loading data set " + data_use +"...")
	if(data_use == 'sign_language'):
		X_train, Y_train, X_test, Y_test, classes = tf_utils.load_dataset_sign_language(False)
		num_logits = 6
	elif(data_use == 'mnist'):
		X_train, Y_train, X_test, Y_test, classes = tf_utils.load_dataset_mnist(False)

		num_train = X_train.shape[1]
		num_test = X_test.shape[1]
		X_train = np.resize(X_train, (28,28,num_train))
		X_test = np.resize(X_test,(28,28,num_test))
		X_train = X_train.T
		X_test = X_test.T
		X_train_3d = np.empty((num_train,28,28, 1), dtype=np.float32)
		X_test_3d = np.empty((num_test,28,28, 1), dtype=np.float32)
		X_train_3d[:,:,:,0]=X_train


		X_test_3d[:,:,:,0]=X_test

		Y_train = Y_train.T
		Y_test = Y_test.T
		X_train = X_train_3d
		X_test = X_test_3d

		num_logits = 10
		'''
		Using layer structure from https://github.com/hwalsuklee/tensorflow-mnist-cnn
		input layer : 784 nodes (MNIST images size)
		first convolution layer : 5x5x32
		first max-pooling layer
		second convolution layer : 5x5x64
		second max-pooling layer
		third fully-connected layer : 1024 nodes
		output layer : 10 nodes (number of class for MNIST)
		'''


	

	'''''''''''''''''''''''''''''''''''''''''''''''
	'			Pre-process data 				  '
	'''''''''''''''''''''''''''''''''''''''''''''''
	

	print ("Pre-processing ...")
	print ("number of training examples = " + str(X_train.shape[0]))
	print ("number of test examples = " + str(X_test.shape[0]))
	print ("X_train shape: " + str(X_train.shape))
	print ("Y_train shape: " + str(Y_train.shape))
	print ("X_test shape: " + str(X_test.shape))
	print ("Y_test shape: " + str(Y_test.shape))



	print (X_train.shape)
	print (Y_train.shape)


	'''''''''''''''''''''''''''''''''''''''''''''''
	'			Specify the structure of network  '
	'''''''''''''''''''''''''''''''''''''''''''''''
	# each layer : conv2d -> relu -> max pool 
	# n_c_prev,n_c
	# conv2d:
	# f,s,padding,
	#
	# max pool:
	# f,padding

	CNN_layer_info={}
	#First layer will c_prev = 3, c =8, f=4, s=1, padding='SAME', max_pool:8x8 filter, padding = 'same', activation :relu
	CNN_layer_info["C1"] = {'layer_type':'CONV_RELU_MAXPOOL','channel':[1,32],'conv2d': [5, 1, 'SAME'],'max_pool': [2, 'VALID'],'activation': 'relu'}
	CNN_layer_info["C2"] = {'layer_type':'CONV_RELU_MAXPOOL','channel':[32,64],'conv2d': [3 ,1, 'SAME'],'max_pool': [2, 'VALID'],'activation': 'relu'}
	#CNN_layer_info["C3"] = {'layer_type':'CONV_RELU_MAXPOOL','channel':[32,120],'conv2d': [2, 1, 'VALID'],'max_pool': [2, 'SAME'],'activation': 'relu'}
	#CNN_layer_info["C4"] = {'layer_type':'CONV_RELU_MAXPOOL','channel':[120,200],'conv2d': [2, 1, 'VALID'],'max_pool': [2, 'SAME'],'activation': 'relu'}
	#CNN_layer_info["C5"] = {'layer_type':'CONV_RELU_MAXPOOL','channel':[64,120],'conv2d': [3, 1, 'VALID'],'max_pool': [2, 'VALID'],'activation': 'relu'}
	#CNN_layer_info["C6"] = {'layer_type':'CONV_RELU_MAXPOOL','channel':[120,200],'conv2d': [3, 1, 'VALID'],'max_pool': [2, 'VALID'],'activation': 'relu'}
	#CNN_layer_info["C3"] = {'layer_type':'CONV_RELU_MAXPOOL','channel':[16,32],'conv2d': [2, 1, 'SAME'],'max_pool': [4, 'SAME'],'activation': 'relu'}
	#CNN_layer_info["C3"] = {'layer_type':'CONV_RELU_MAXPOOL','channel':[64,80],'conv2d': [3, 1, 'SAME'],'max_pool': [2, 'SAME'],'activation': 'relu'}

	CNN_layer_info["C3"] = {'layer_type':'FLATTEN'}
	#CNN_layer_info["C4"] = {'layer_type':'FULLY_CONNECTED', 'neuron_size':300}
	CNN_layer_info["C4"] = {'layer_type':'FULLY_CONNECTED', 'neuron_size':1020}
	CNN_layer_info["C5"] = {'layer_type':'FULLY_CONNECTED', 'neuron_size':num_logits}
	#CNN_layer_info["C5"] = {'layer_type':'FULLY_CONNECTED', 'neuron_size':num_logits}
	#parameters=tf_CNN_wrapper.initialize_parameters(CNN_layer_info)
	#print (parameters)



	'''''''''''''''''''''''''''''''''''''''''''''''
	'			Train the model					  '
	'''''''''''''''''''''''''''''''''''''''''''''''

	#Specify number of layers

	print("Training model ...")
	

	#Train the model
	learning_rate = 0.0001
	num_epoch = 100
	minibatch_size = 120

	train_accuracy, test_accuracy, parameters,costs = tf_CNN_wrapper.model(
		X_train, 
		Y_train, 
		X_test, 
		Y_test,
		CNN_layer_info, 
		learning_rate,    #learing rate
		num_epoch, #num epoch
		minibatch_size , #	minibatch_size 
		True #print cost
		)
	if(1):
		#write the parameters to file
		np.save('parameters/parametersCNN'+data_use+'.npy',parameters)

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
	if((NUMPY_EXIST==False )| (TENSORFLOW_EXIST == False)):
		if(NUMPY_EXIST==False):
			print ("NUMPY DOES NOT EXIST >>> PLEASE INSTALL " )
		if(TENSORFLOW_EXIST==False):
			print("TENSORFLOW DOES NOT EXIST >>> PLEASE INSTALL")
	else:
		main()