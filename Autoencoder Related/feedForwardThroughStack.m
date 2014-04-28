function [a] = feedForwardThroughStack(stackedTheta, X)

%% Computes upper layer activation, which can be used as input to classification task

% Credits:

% Dependencies:
% 1. function library

% Assists:

% Future mods:

% Notes:
% 1. X:             Matrix containing the training data, instances in rows
% 2. Stackedtheta:  Stack containing layers parameters

% -------------------------------------------------------------------------

activation    = cell(numel(stackedTheta)+1, 1);    % Including input, number of activations = number of theta layers + 1
activation{1} = X;

for i = 2:(numel(stackedTheta)+1)
    z = bsxfun(@plus, activation{i-1}*stackedTheta{i-1}.w, stackedTheta{i-1}.b);      % in this case, w1 corresponds to layers 1 and 2, and so on (unfortunate convention, can change going forward)
    activation{i} = sigmoid(z);
end

a = activation{end};

end
