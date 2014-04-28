function [X] = doPCAProcessing(X, U, S)
%% Not Validated
%% Script to perform pre-processing corresponding to PCA WHITENING on an original
% data matrix(mean normalized data matrix with examples in rows and featueres in columns)
% For example, if PCA has been performed on XTrain, this can perform that bit
% of pre-processing on XTest corresponding to PCA on XTrain
% Input required is data matrix, left singular vector matrix U and singular
% value matrix S

% Performs PCA rotation and scaling based on U & S, U can be dimensionally
% reduced as required

% Future mods:

% Dependencies:
% 1. Matlab built in functions - cov, svd, diag, sqrt, ones, size

% Refer PCA exercise in UFLDL Tutorial
epsilon = 1e-5;                                 % Regularization
X       = X*U;                                  % Rotated (considering column vectors, u'x is performed but since x is row and u is column, this works just the same
X       = X*diag(1./sqrt(diag(S) + epsilon));   % normalize as in PCA whitening (refer UFLDL exercise)

end