import cv2
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from tf_wrappers import forward_propagation
import tensorflow as tf
import numpy as np

cap = cv2.VideoCapture(0)
parameters = np.load('parameters.npy').item()



x = tf.placeholder("float", [12288, 1])

layer_array= [12288,1000,100,6]
sess = tf.Session()
    

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

print (X_test_orig.shape)
while(True):
  if(0):
    ret,frame = cap.read()
    frame = cv2.resize(frame,(64,64), interpolation = cv2.INTER_CUBIC)
    frame_array = np.reshape(frame,(12288,1))
    frame_array=frame_array/255
    z3 = forward_propagation(x, parameters,layer_array)
    p = tf.argmax(z3)
    #my_image_prediction = predict(frame_array, parameters)
    #print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
    frame_array=frame_array
    prediction = sess.run(p, feed_dict = {x: frame_array})
    print("Your algorithm predicts: y = " + str(np.squeeze(prediction)))
    cv2.imshow("a",frame)
    cv2.waitKey(1)
  else:
    for test_ in X_test_orig:
      
      
      frame = cv2.resize(test_,(64,64), interpolation = cv2.INTER_CUBIC)
      frame_array = np.reshape(frame,(12288,1))
      frame_array=frame_array/255
      z3 = forward_propagation(x, parameters,layer_array)
      #z3 = forward_propagation(x, parameters,layer_array)
      p = tf.argmax(z3)
      #my_image_prediction = predict(frame_array, parameters)
      #print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
      frame_array=frame_array
      prediction = sess.run(p, feed_dict = {x: frame_array})
      cv2.putText(test_,str(prediction[0]),(10,10), 1,1,(0,255,255),2,cv2.LINE_AA)
      cv2.imshow("test",test_)
      cv2.waitKey(100)



# if(0):
# np.save('parameters.npy',parameters)

# import scipy
# from PIL import Image
# from scipy import ndimage

# ## START CODE HERE ## (PUT YOUR IMAGE NAME) 
# my_image = "thumbs_up.jpg"
# ## END CODE HERE ##

# # We preprocess your image to fit your algorithm.
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
# my_image_prediction = predict(my_image, parameters)

# plt.imshow(image)
# print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))


# # You indeed deserved a "thumbs-up" although as you can see the algorithm seems to classify it incorrectly. The reason is that the training set doesn't contain any "thumbs-up", so the model doesn't know how to deal with it! We call that a "mismatched data distribution" and it is one of the various of the next course on "Structuring Machine Learning Projects".

# # <font color='blue'>
# # **What you should remember**:
# # - Tensorflow is a programming framework used in deep learning
# # - The two main object classes in tensorflow are Tensors and Operators. 
# # - When you code in tensorflow you have to take the following steps:
# #     - Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...)
# #     - Create a session
# #     - Initialize the session
# #     - Run the session to execute the graph
# # - You can execute the graph multiple times as you've seen in model()
# # - The backpropagation and optimization is automatically done when running the session on the "optimizer" object.
