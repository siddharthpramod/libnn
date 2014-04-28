function sigm = sigmoid(x)
%% Implementation of the sigmoid (logistic) function 
    sigm = 1 ./ (1 + exp(-x));
end