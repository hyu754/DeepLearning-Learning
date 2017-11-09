# TensorFlow mini problem
This repo uses Andrew Ng's tensorflow [tutorial code](deeplearning.ai) for a general network.

Most of the functions/utilities are from Andrew Ng's tutorials, I have just modified it to fit a general network with <img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" align=middle width="11.14542pt" height="22.38192pt"/> layers, and the size of each layer <img alt="$n^[l]$" src="svgs/835af22ecc819595a832d512de4ddbe1.svg" align=middle width="24.13257pt" height="29.12679pt"/>

## Cost
We use the cost function 

<p align="center"><img alt="$$ J = - \frac{1}{m}  \sum_{i = 1}^m  \large ( \small y^{(i)} \log a^{ [2] (i)} + (1-y^{(i)})\log (1-a^{ [2] (i)} )\large )\small\tag{2}$$" src="svgs/4ebb7af73bc02d7114f54fbbf184538c.svg" align=middle width="381.0675pt" height="44.878845pt"/></p>

## One hot
The code uses one hot encoding, such that the vector <img alt="$[1,2,3,0,2,1]$" src="svgs/6c0f8e8e7ec24f899b68a9dc242496ce.svg" align=middle width="94.673535pt" height="24.56553pt"/>, will become:

<p align="center"><img alt="$$&#10;M=  \begin{bmatrix}&#10;    0 &amp; 0 &amp; 0&amp;1&amp;0&amp;0\\&#10;    1 &amp; 0 &amp; 0&amp;0&amp;0&amp;1\\&#10;&#9;0 &amp; 1 &amp; 0&amp;0&amp;1&amp;0\\&#10;&#9;0 &amp; 0 &amp; 1&amp;0&amp;0&amp;0\\&#10;  \end{bmatrix}&#10;$$" src="svgs/2b0f3a9a55890e3d077247d126a62837.svg" align=middle width="192.9345pt" height="78.794265pt"/></p>