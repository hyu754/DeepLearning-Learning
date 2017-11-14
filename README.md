# TensorFlow mini problem
This repo uses Andrew Ng's tensorflow [tutorial code](deeplearning.ai) for a range of vanilla networks.

## Deep Neural Network (DNN)

Most of the functions/utilities are from Andrew Ng's tutorials, I have just modified it to fit a general network; where hyperparameters can be modified, namely, the number of layers (<img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" align=middle width="11.14542pt" height="22.38192pt"/>), and the size of each layer (<img alt="$n^{[l]}$" src="svgs/8242f44bc9e80233af0d6944ea868001.svg" align=middle width="21.453465pt" height="29.12679pt"/>).



### Cost
We use the cost function 

<p align="center"><img alt="$$ J = - \frac{1}{m}  \sum_{i = 1}^m  \large ( \small y^{(i)} \log a^{ [2] (i)} + (1-y^{(i)})\log (1-a^{ [2] (i)} )\large )\small\tag{2}$$" src="svgs/4ebb7af73bc02d7114f54fbbf184538c.svg" align=middle width="381.0675pt" height="44.878845pt"/></p>

### One hot
The code uses one hot encoding, such that the vector <img alt="$[1,2,3,0,2,1]$" src="svgs/6c0f8e8e7ec24f899b68a9dc242496ce.svg" align=middle width="94.673535pt" height="24.56553pt"/>, will become:

<p align="center"><img alt="$$&#10;M=  \begin{bmatrix}&#10;    0 &amp; 0 &amp; 0&amp;1&amp;0&amp;0\\&#10;    1 &amp; 0 &amp; 0&amp;0&amp;0&amp;1\\&#10;&#9;0 &amp; 1 &amp; 0&amp;0&amp;1&amp;0\\&#10;&#9;0 &amp; 0 &amp; 1&amp;0&amp;0&amp;0\\&#10;  \end{bmatrix}&#10;$$" src="svgs/2b0f3a9a55890e3d077247d126a62837.svg" align=middle width="192.9345pt" height="78.794265pt"/></p>



## Convolution Neural Network (CNN)

Lets have a look at some of the parameters/formulats. 

-<img alt="$n_H^{l},n_W^{l},n_C^{l}$" src="svgs/c9b4154aedf24325855c2df7bb0f0a76.svg" align=middle width="81.686385pt" height="27.85299pt"/> - the height, width and number of channels for a given layer, <img alt="$l$" src="svgs/2f2322dff5bde89c37bcae4116fe20a8.svg" align=middle width="5.2088685pt" height="22.74591pt"/> .

-The output shape of a convolution regarding the input shape can be written as


<p align="center"><img alt="$$n_H = \lfloor{\frac{n_H_{prev} - f + 2 \ times pad}{stride} \rfloor} +1$$" src="svgs/20dbee70e1cfe02a61f724a0545be98b.svg" align=middle width="271.1511pt" height="33.769395pt"/></p>

<p align="center"><img alt="$$n_W = \lfloor{\frac{n_W_{prev} - f + 2 \ times pad}{stride} \rfloor} +1$$" src="svgs/d062206edb5f1939cdc17d0880587e9e.svg" align=middle width="276.1242pt" height="33.769395pt"/></p>





