% function [] = checkStackedAECost()

% Check the gradients for an autoencoder
%
% Adapted from UFLDL, good for 5 hidden layer stacks
%

%% Setup random data / small model
%  This is starter code to check gradients of autoencoder cost function,
%  edit as required

clc;

inputSize  = 10;
hiddenSize = 5;
lambda = 0.3;
lambdaSup = 0.3;
beta = 3;
betaSup = 2;
sparsityParam = 0.1;

data   = randn(10, inputSize);
X      = randn(7, inputSize);   % For partially supervised
y      = [0 1 1 0 1 0 0]';



theta = initializeAutoencoderParameters(inputSize, hiddenSize);
thetaSup = 0.005 * randn(hiddenSize+1, 1);       % +1 to include intercept term

fullTheta = [ theta(:) ; thetaSup(:) ];

[cost, grad] = partiallySupervisedSparseAutoencoderLogLoss(fullTheta, inputSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data, X, y, lambdaSup, betaSup);

% Check that the numerical and analytic gradients are the same
numgrad = computeNumericalGradient( @(p) partiallySupervisedSparseAutoencoderLogLoss(p, inputSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data, X, y, lambdaSup, betaSup), fullTheta);
                     
% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
disp('Norm between numerical and analytical gradient (should be less than 1e-9)');
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

            % When you got this working, Congratulations!!! 
            
            
