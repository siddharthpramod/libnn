function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)
%% This script is sourced from UFLDL Tutorial

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);      % 10x784

numCases = size(data, 2);                           % 60000

groundTruth = full(sparse(labels, 1:numCases, 1));  % 10x60000
cost = 0;

thetagrad = zeros(numClasses, inputSize);           % 10x784

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

X = data;                       % 784x60000

%Next two lines used to prevent overflow in computing exponential values
M = theta*X;                    % 10x60000
M = bsxfun(@minus, M, max(M, [], 1));

h = exp(M);                     % 10x60000
h = bsxfun(@rdivide,h,sum(h));

cost = -1/numCases*sum(sum(groundTruth.*log(h))) + lambda/2*sum(sum(theta.^2));  %sum(10x60000 .* 10x60000)

thetagrad = -1/numCases*(groundTruth-h)*X' + lambda*theta; %10x784 = 10x60000*60000x784

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

