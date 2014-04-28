function [J, grad] = logRegCostFunction(theta, X, y, lambda)
%% COSTFUNCTION Compute cost and gradient for logistic regression
%  Warning: Intercept term should be pre-included in data matrix & theta
%  vector as the first column & first term respectively

% Credits:

% Dependencies:
% 1. function library (or sigmoid functions script)

% Assists:

% Future mods:
% 1. mod for options

% Notes:
% 1. theta is a column vector of initialized parameters including the bias
%    term ( (n+1) x 1 )
% 2. X is data matrix containing examples in rows including the "1" for
%    intercept term (m x (n+1))
% 3. y is column vector for expected activation for each example (m x 1)
% 4. Intercept term in data & theta are first column and first term
%    respectively. This matters when considering regularization

% Initialize some useful values
m    = length(y);       % number of training examples
    
thetaReg    = theta;    % omit intercept term for regularization
thetaReg(1) = 0;

h    = sigmoid(X*theta); 

J    = sum(-y.*log(h)-(1-y).*log(1-h))/m      + lambda/2/m*sum(thetaReg.^2);
grad = (sum(((h-y)*ones(1,size(X,2))).*X)/m)' + lambda/m*(thetaReg);

% =============================================================

end
