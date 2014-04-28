function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
%% Computes cost (reconstruction error) and gradient for one layer autoencoder
% Implements a linear decoder
% Validated against one generic trial, can use more but not necessary

% Credits:
% 1. Adapted from sparse autoencoder exercise (UFLDL tutorial:
%    http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)

% Dependencies:
% 1. function library

% Assists:

% Future mods:
% 1. mod for options between logistic and tanh function

% Notes:
% 1. visibleSize:   the number of input units  
% 2. hiddenSize:    the number of hidden units 
% 3. lambda:        weight decay parameter
% 4. sparsityParam: The desired average activation for the hidden units
% 5. beta:          weight of sparsity penalty term

% 6. data:          Matrix containing the training data, examples in rows
% 7. theta:         Vector containing W1, W2, b1, b2.
%                   W1 is (visible x hidden), W2 is (hidden x visible)

W1 = reshape(theta(1:hiddenSize*visibleSize),                          visibleSize, hiddenSize);    % visible x hidden
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), hiddenSize, visibleSize);    % hidden  x visible
b1 = theta  (2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize)';                      % 1       x hidden
b2 = theta  (2*hiddenSize*visibleSize+hiddenSize+1:end)';                                           % 1       x visible

cost   = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

m  = size(data,1);

z2 = bsxfun(@plus, data*W1, b1);    % m x hidden
a2 = sigmoid(z2);

a3 = bsxfun(@plus, a2*W2, b2);      % m x visible

rho           = sparsityParam * ones(1,hiddenSize); % 1 x hidden
rho_hat       = mean(a2);                           % 1 x hidden
KL_penalty    = sum (rho.*log(rho./rho_hat) + (1-rho).*log((1-rho)./(1-rho_hat)));
sparsity_term = beta*(-rho./rho_hat + (1-rho)./(1-rho_hat));   % 1 x hidden

delta3 = -(data-a3);                                                        % m x visible
delta2 = bsxfun(@plus, delta3*W2', sparsity_term).*sigmoidDerivative(a2);   % m x hidden

cost   = 1/2/m*sum(sum((a3-data).^2)) + lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2))) + beta*KL_penalty;   % half-square error term + regularization + sparsity penalty

W1grad = data'*delta2/m + lambda*W1;    % visible x hidden
W2grad = a2'  *delta3/m + lambda*W2;    % hidden  x visible
b1grad = mean(delta2);                  % 1 x hidden
b2grad = mean(delta3);                  % 1 x visible

% Convert the gradients back to a vector format (suitable for minFunc)
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end