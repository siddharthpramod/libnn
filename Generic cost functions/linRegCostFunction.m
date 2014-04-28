function [J, grad] = linRegCostFunction(theta, X, y, lambda)
%% COSTFUNCTION Compute cost and gradient for linear regression
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

m = length(y);          % number of training examples

thetaReg    = theta;    % omit intercept term for regularization
thetaReg(1) = 0;

h    = X*theta;

J    = 1/2/m*sum((h-y).^2) + lambda/2/m*sum(thetaReg.^2);
grad = 1/m*(X'*(h-y))      + lambda/m*thetaReg;

end
