% function [] = checkStackedAECost()

% Check the gradients for the stacked autoencoder
%
% Adapted from UFLDL, good for 5 hidden layer stacks
%

%% Setup random data / small model

clc;

L0Size = 10;
L1Size = 10;
L2Size = 9;
L3Size = 8;
L4Size = 7;
L5Size = 6;
L6Size = 5;
L7Size = 4;
lambda = 0.01;
X      = randn(7, L0Size);
y      = [0 1 1 0 1 0 0]';

stack = cell(5,1);
stack{1}.w = 0.1 * randn(L0Size, L1Size);
stack{1}.b = zeros(1, L1Size);
stack{2}.w = 0.1 * randn(L1Size, L2Size);
stack{2}.b = zeros(1, L2Size);
stack{3}.w = 0.1 * randn(L2Size, L3Size);
stack{3}.b = zeros(1, L3Size);
stack{4}.w = 0.1 * randn(L3Size, L4Size);
stack{4}.b = zeros(1, L4Size);
stack{5}.w = 0.1 * randn(L4Size, L5Size);
stack{5}.b = zeros(1, L5Size);
stack{6}.w = 0.1 * randn(L5Size, L6Size);
stack{6}.b = zeros(1, L6Size);
stack{7}.w = 0.1 * randn(L6Size, L7Size);
stack{7}.b = zeros(1, L7Size);

logRegTheta = 0.005 * randn(L7Size+1, 1);       % +1 to include intercept term

[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ logRegTheta ; stackparams ];


[cost, grad] = stackCost(stackedAETheta, X, y, lambda, netconfig);

% Check that the numerical and analytic gradients are the same
numgrad = computeNumericalGradient( @(p) stackCost(p, X, y, lambda, netconfig), stackedAETheta);

% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
disp('Norm between numerical and analytical gradient (should be less than 1e-9)');
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

            % When you got this working, Congratulations!!! 
            
            
