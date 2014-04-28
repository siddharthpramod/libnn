libnn
=====

A library of Matlab routines for deeplearning


Most of the routines in this library are adapted from the UFLDL Tutorial located at:
http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial

The routines also depend on Minfunc for optimization, available for download at:
http://www.di.ens.fr/~mschmidt/Software/copyright.html

Description of the folders:

1. Activation Functions:
These are implementations of the sigmoid (logistic), it's "derivative" as may be used in 
the deep learning setting, and the tanh "derivative.

2. Autoencoder Related:
These are what will be of primary use.
(a) sparseAutoencoderCost.m is used to find the cost of an autoencoder trained using a 
sparsity constraint.
(b) sparseAutoencoderLinearCost.m is similar to (a) but implements a linear decoder. More 
information about both these methods is available in the UFLDL tutorial.
(c) partiallySupervisedAutoencoderCost.m is similar to (a) but implements a joint 
objective - the autoencoder reconstruction cost plus a supervised objective, 
a technique described in Section 6 of the 2007 paper on "Greedy Layer-Wise Training of 
Deep Networks" by Bengio et. al., to deal with "uncooperative input distributions".

Also includes a useful utility - analyzeAutoencoder.m that provides an f1 measure and 
plots the reconstruction against the input, to analyze learned encodings.

3. Evaluations:
Currently includes an implementation to find the F1 score

4. Generic Cost Functions:
To find cost and gradient for linear, logistic and softmax regression

5. Whitening:
Implementations of PCA and ZCA whitening. "Full" implies it performs SVD on the provided
data and returns transformed data, the left singular vector matrix and the singular value
(diagonal) matrix. Useful when supplying training data.
"Processing" implies it performs only transformation, given the data, the left singular 
matrix and the singular value matrix (the latter two obtained from the "Full" operation).
Useful when supplying test data.
