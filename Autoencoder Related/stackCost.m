function [cost,grad] = stackCost(theta, X, y, lambda, netconfig)

%% Computes cost and gradient for a stacked neural network for a final binary classification using logistic regression
% Validated using checkStackedAECost.m for 5 layer stack

% Credits:
% 1. Adapted from sparse autoencoder exercise (UFLDL tutorial:
%    http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)

% Dependencies:
% 1. function library

% Assists:

% Future mods:
% 1. Add options for different final layer functions
% 2. Optimize - get rid of loops

% Notes:
% 1. X:             Matrix containing the training data, examples in rows
% 2. y:             Target output
% 3. lambda:        Regularization parameter
% 4. theta:         Vector containing all parameters
% 5. netconfig:     network configuration required to convert parameter
%                   vector to stack
% 
% layer sizes can be inferred from stack dimensions or from netconfig



% -------------------------------------------------------------------------

% Unroll parameter vector

LEndSize = netconfig.layersizes{end};

logRegTheta = reshape(theta(1:(LEndSize+1)), (LEndSize+1), 1);      % LEndSize '+ 1' because intercept term included.

% Extract the "stack"
stack = params2stack(theta(LEndSize+2:end), netconfig);

% Initialize Parameters

cost = 0;

logRegThetaGrad = zeros(size(logRegTheta));

stackgrad = cell(size(stack));

for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

% -------------------------------------------------------------------------

m  = numel(y);

activation    = cell(numel(stack)+1, 1);
activation{1} = X;

for i = 2:(numel(stack)+1)
    z = bsxfun(@plus, activation{i-1}*stack{i-1}.w, stack{i-1}.b);      % in this case, w1 corresponds to layers 1 and 2, and so on (unfortunate convention, can change going forward)
    activation{i} = sigmoid(z);
end

z = [ones(m, 1) activation{end}];   % m x 1+LEndSize
h = sigmoid(z*logRegTheta);         % m x 1

% delta cells: delta{3} corresponds to delta3
% delta{1} does not exist

delta = cell(numel(stack) + 1, 1);
delta{end} = ((h-y)*logRegTheta(2:end)').*sigmoidDerivative(activation{end});  % theta(2:end) because intercept term to be omitted
for i = (numel(stack)):-1:2
    delta{i} = (delta{i+1}*(stack{i}.w)').*sigmoidDerivative(activation{i});
end

% -------------------------------------------------------------------------

thetaReg    = logRegTheta;    % omit intercept term for regularization
thetaReg(1) = 0;

cost = sum(-y.*log(h)-(1-y).*log(1-h))/m + lambda/2/m*sum(thetaReg.^2); % Used log reg cost function

logRegThetaGrad = (sum(((h-y)*ones(1,LEndSize+1)).*[ones(m,1) activation{end}])/m)' + lambda/m*(thetaReg); 

for i = 1:numel(stackgrad)
    stackgrad{i}.w = activation{i}'*delta{i+1}/m;
    stackgrad{i}.b = mean(delta{i+1});
end

% Convert the gradients back to a vector format (suitable for minFunc)
grad = [logRegThetaGrad(:) ; stack2params(stackgrad)];

end
