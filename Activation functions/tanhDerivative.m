function tanh_prime = tanhDerivative(a)
%% Implementation of the derivative of tanh function given activation 'a'
%  To use in backpropagation when activations have been computed during
%  forward propagation
    tanh_prime = 1-a.^2;
end