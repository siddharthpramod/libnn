function [activation] = feedForwardAutoencoder(theta, visibleSize, hiddenSize, data)

%% This script is sourced from UFLDL exercise

% Dependencies:
% 1. Function Library

% theta: trained weights from the autoencoder
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
  
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), visibleSize, hiddenSize);         % visible x hidden
b1 = theta  (2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize)';  % 1       x hidden

m  = size(data,1);

z2 = bsxfun(@plus, data*W1, b1);    % m x hidden
activation = sigmoid(z2);

%-------------------------------------------------------------------

end
