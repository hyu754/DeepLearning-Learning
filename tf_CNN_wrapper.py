import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import  random_mini_batches_CNN
import numpy as np



def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(tf.float32,shape = (None,n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32,shape = (None, n_y))
    ### END CODE HERE ###
    
    return X, Y


def initialize_parameters(CNN_layers):
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
    parameters ={}
    num_layers = len(CNN_layers)
    for l in range(0,num_layers):
        #Only need to initialize for following types of layers
        if(CNN_layers["C"+str(l+1)]['layer_type'] == 'CONV_RELU_MAXPOOL'):
            conv2d_info = CNN_layers["C"+str(l+1)]["conv2d"]
            channel_info = CNN_layers["C"+str(l+1)]["channel"]
            f = conv2d_info[0] 
            c_prev = channel_info[0]
            c = channel_info[1]

            parameters["W"+str(l+1)] = tf.get_variable("W"+str(l+1),[f,f,c_prev,c],initializer = tf.contrib.layers.xavier_initializer(seed =0),dtype = tf.float32)
    ### START CODE HERE ### (approx. 2 lines of code)
    #3W1 = tf.get_variable("W1",[4,4,3,8],initializer = tf.contrib.layers.xavier_initializer(seed =0),dtype = tf.float32)
    #W2 = tf.get_variable("W2",[2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed=0),dtype = tf.float32)
    ### END CODE HERE ###

    #parameters = {"W1": W1,"W2": W2}
    
    return parameters



def forward_propagation(X, parameters,CNN_layers):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 

    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z_prev =None
    A_prev = None
    P_prev = None
    num_layer = len(CNN_layers)
    for l in range(0,num_layer):
        #if conv->relu->maxpool
        if(CNN_layers["C"+str(l+1)]['layer_type'] == 'CONV_RELU_MAXPOOL'):
            conv2d_info = CNN_layers["C"+str(l+1)]["conv2d"] 
            maxpool_info = CNN_layers["C"+str(l+1)]["max_pool"]
            channel_info = CNN_layers["C"+str(l+1)]["channel"]
            print ("initializing conv_relu")
            conv2d_s = conv2d_info[1]
            maxpool_f = maxpool_info[0]

            conv2d_pad = conv2d_info[2]
            maxpool_pad = maxpool_info[1]
            if(l==0):
                Z_prev = tf.nn.conv2d(X,parameters["W"+str(l+1)],strides =[1,conv2d_s,conv2d_s,1],padding = conv2d_pad)

                A_prev = tf.nn.relu(Z_prev)

                P_prev = tf.nn.max_pool(A_prev,ksize = [1,maxpool_f,maxpool_f,1],strides = [1,maxpool_f,maxpool_f,1],padding = maxpool_pad)
            else:
                Z_prev = tf.nn.conv2d(P_prev,parameters["W"+str(l+1)],strides =[1,conv2d_s,conv2d_s,1],padding = conv2d_pad)

                A_prev = tf.nn.relu(Z_prev)

                P_prev = tf.nn.max_pool(A_prev,ksize = [1,maxpool_f,maxpool_f,1],strides = [1,maxpool_f,maxpool_f,1],padding = maxpool_pad)
        elif(CNN_layers["C"+str(l+1)]['layer_type'] == 'FLATTEN'):

            if(l>0):
                #if previous layer WAS CONV->RELU->MAXPOOL
                if(CNN_layers["C"+str(l)]['layer_type'] == 'CONV_RELU_MAXPOOL'):
                    P_prev = tf.contrib.layers.flatten(P_prev)

        elif(CNN_layers["C"+str(l+1)]['layer_type'] == 'FULLY_CONNECTED'):
            if(CNN_layers["C"+str(l)]['layer_type'] == 'FLATTEN'):
                num_neuron = CNN_layers["C"+str(l+1)]['neuron_size']
                Z_prev = tf.contrib.layers.fully_connected(P_prev,num_neuron,activation_fn=None)
            elif(CNN_layers["C"+str(l)]['layer_type'] == 'FULLY_CONNECTED'):
                num_neuron = CNN_layers["C"+str(l+1)]['neuron_size']
                Z_prev = tf.contrib.layers.fully_connected(Z_prev,num_neuron,activation_fn=None)

    #  # CONV2D: filters W2, stride 1, padding 'SAME'
    # Z2 = tf.nn.conv2d(P1,W2,strides =[1,1,1,1],padding = 'SAME')
    # # RELU
    # A2 = tf.nn.relu(Z2)
    # # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding = 'SAME')
    # # FLATTEN
    
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    
    ### END CODE HERE ###

   

    return Z_prev

def compute_cost(Z, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
    ### END CODE HERE ###
    
    return cost


def model(X_train, Y_train, X_test, Y_test,CNN_layers, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    if(len(X_train.shape) == 4):
        (m, n_H0, n_W0, n_C0) = X_train.shape  
    elif(len(X_train.shape) == 3):
        (m, n_H0, n_W0) = X_train.shape   
        n_C0=1     
    n_y = Y_train.shape[1]       

    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters(CNN_layers)
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    #print (parameters)
    Z3 = forward_propagation(X, parameters,CNN_layers)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)

    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer =tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches_CNN(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                #print (minibatch_X.shape,minibatch_Y.shape)
                #print (X,Y)
                _ , temp_cost =  sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        parameters = sess.run(parameters)
        return train_accuracy, test_accuracy, parameters,costs

