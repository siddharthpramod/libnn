function [W1, W2, b1, b2] = unrollAutoencoderTheta(theta, visibleSize, hiddenSize)

%% Script to unroll theta vector into sparse autoencoder weights
% Sourced from UFLDL

% visibleSize is input layer size, hiddenSize is encoded layer size

W1 = reshape(theta(1:hiddenSize*visibleSize),                          visibleSize, hiddenSize);    % visible x hidden
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), hiddenSize, visibleSize);    % hidden  x visible
b1 = theta  (2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize)';                      % 1       x hidden
b2 = theta  (2*hiddenSize*visibleSize+hiddenSize+1:end)';                                           % 1       x visible

end