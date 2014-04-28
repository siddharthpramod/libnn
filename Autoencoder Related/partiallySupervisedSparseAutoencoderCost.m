function [cost,grad] = partiallySupervisedSparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data, X, y, lambdaSup, betaSup)
%% Computes cost (reconstruction error + prediction ) and gradient for one layer autoencoder
%  Validated using checkAutoencoderCost.m

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
% 7. theta:         Vector containing W1, W2, b1, b2, W2Sup.
%                   W1 is (visible x hidden), W2 is (hidden x visible)
%                   W2Sup is (1 + hidden) for logistic regression
% 8. X, y:          Supervised training data + labels   

W1 = reshape(theta(1:hiddenSize*visibleSize),                          visibleSize, hiddenSize);    % visible x hidden
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), hiddenSize, visibleSize);    % hidden  x visible
b1 = theta  (2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize)';                      % 1       x hidden
b2 = theta  (2*hiddenSize*visibleSize+hiddenSize+1:2*hiddenSize*visibleSize+hiddenSize+visibleSize)';  % 1    x visible
W2Sup = theta (2*hiddenSize*visibleSize+hiddenSize+visibleSize+1:end);                             % 1+hidden x 1

cost   = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));
W2Supgrad = zeros(size(W2Sup));

% -------------------------------------------------------------------------

m  = size(data, 1);
mSup = size(X, 1);

z2 = bsxfun(@plus, data*W1, b1);    % m x hidden
a2 = sigmoid(z2);

z3 = bsxfun(@plus, a2*W2, b2);      % m x visible
a3 = sigmoid(z3);

z2Sup = bsxfun(@plus, X*W1, b1);    % mSup x hidden
a2Sup = sigmoid(z2Sup);

h = sigmoid([ones(numel(y), 1) a2Sup]*W2Sup); % mSup x 1

% -------------------------------------------------------------------------

rho           = sparsityParam * ones(1,hiddenSize); % 1 x hidden
rho_hat       = mean(a2);                           % 1 x hidden
KL_penalty    = sum (rho.*log(rho./rho_hat) + (1-rho).*log((1-rho)./(1-rho_hat)));
sparsity_term = beta*(-rho./rho_hat + (1-rho)./(1-rho_hat));   % 1 x hidden

delta3 = (a3-data).*sigmoidDerivative(a3);                                  % m x visible
delta2 = bsxfun(@plus, delta3*W2', sparsity_term).*sigmoidDerivative(a2);   % m x hidden

delta3Sup = h - y;                                              % mSup x 1
delta2Sup = delta3Sup*W2Sup(2:end)'.*sigmoidDerivative(a2Sup);  % mSup x hidden; 2:end to remove intercept term

% -------------------------------------------------------------------------

cost   = 1/2/m*sum(sum((a3-data).^2)) + lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2))) + beta*KL_penalty ...
    + betaSup*(sum(-y.*log(h)-(1-y).*log(1-h))/mSup + lambdaSup/2*(sum(W2Sup(2:end).^2)));

% half-square error term + regularization + sparsity penalty ...
% + supervisedWeight*(log loss error + regularization)

% -------------------------------------------------------------------------

W1grad = (data'*delta2/m + lambda*W1) + betaSup*(X'*delta2Sup/mSup);  % visible x hidden
W2grad =  a2'  *delta3/m + lambda*W2;                                   % hidden  x visible
b1grad = mean(delta2)    + betaSup*mean(delta2Sup);                     % 1 x hidden
b2grad = mean(delta3);                                                  % 1 x visible
W2Supgrad = betaSup*[mean(delta3Sup); (a2Sup'*delta3Sup/mSup + lambdaSup*W2Sup(2:end))]; % 1+hidden x 1

% Convert the gradients back to a vector format (suitable for minFunc)
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:); W2Supgrad(:)];

end
