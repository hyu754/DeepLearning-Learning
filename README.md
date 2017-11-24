<!--python -m readme2tex --output README.md OTHEREADME.md --nocdn -->
# TensorFlow mini problem
This repo uses Andrew Ng's tensorflow [tutorial code](deeplearning.ai) for a range of vanilla networks.

## Deep Neural Network (DNN)

Most of the functions/utilities are from Andrew Ng's tutorials, I have just modified it to fit a general network; where hyperparameters can be modified, namely, the number of layers (<img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" align=middle width="11.14542pt" height="22.38192pt"/>), and the size of each layer (<img alt="$n^{[l]}$" src="svgs/8242f44bc9e80233af0d6944ea868001.svg" align=middle width="21.453465pt" height="29.12679pt"/>).

To train a model, make your own `load_data()` function in `tf_utils.py`.

After, to train your network, open `train_NN.py`. 

### Things to change
-Data loading function
```python

def load_data():
	#Load your data
	return train_X, train_Y, test_X, test_Y, classes
```

-Layers information, the first and last layers' dimension is already defined. Below 'layer_array' specifies 3 Hidden units, with 20,10 and 10, neurons in each layer.
```python
layer_array= [X_train.shape[0],20,10,10,Y_train.shape[0]]
```

-Specify tunable parameters
```python
learning_rate = 0.00005
num_epoch =10000
minibatch_size = 32
```



## Convolution Neural Network (CNN)

Let's have a look at some of the parameters/formulats. 

-<img alt="$n_H^{[l]},n_W^{[l]},n_C^{[l]}$" src="svgs/f52c9f937e82f5aa3bbe4621962e8d2d.svg" align=middle width="83.134095pt" height="34.27314pt"/> - the height, width and number of channels for a given layer, <img alt="$l$" src="svgs/2f2322dff5bde89c37bcae4116fe20a8.svg" align=middle width="5.2088685pt" height="22.74591pt"/> .

-The output shape of a convolution regarding the input shape can be written as


<p align="center"><img alt="$$n_H = \lfloor{\frac{n_H_{prev} - f + 2 \times pad}{stride} \rfloor} +1$$" src="svgs/88eddbdc741c202ee16bb6a00ff4952e.svg" align=middle width="244.3716pt" height="33.769395pt"/></p>

<p align="center"><img alt="$$n_W = \lfloor{\frac{n_W_{prev} - f + 2 \times pad}{stride} \rfloor} +1$$" src="svgs/2be0fd5596580ecfa71b87f9370ae580.svg" align=middle width="249.3447pt" height="33.769395pt"/></p>


.............


