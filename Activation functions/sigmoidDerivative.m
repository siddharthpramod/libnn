function sig_prime = sigmoidDerivative(a)
%% Implementation of the derivative of sigmoid (logistic) function given activation 'a'
%  To use in backpropagation when activations have been computed during
%  forward propagation
    sig_prime = a.*(1-a);
end