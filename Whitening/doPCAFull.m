function [X, U, S] = doPCAFull(X)
%% VALIDATED (SORT OF, ONLY VISUALLY AND FOUND ~0 COVARIANCE)
%% Script to perform PCA WHITENING on X (mean normalized data matrix with examples in rows and featueres in columns)
% Returns PCA WHITENED features, left singular vector matrix U and diagonal
% matrix S, to use for PCA whitening in other data matrices such as test &
% validation matrices that require same processing as the training matrix

% Performs PCA including determining covariance matrix, performs SVD,
% rotation based on left singular vectors and normalizing to unit variance based on singular
% values. ZCA can be performed using left singular vectors and implemented
% as another function.

% Future mods:
% 1. Option for PCA dimensionality reduction

% Dependencies:
% 1. Matlab built in functions - cov, svd, diag, sqrt, ones, size

% Refer PCA exercise in UFLDL Tutorial
epsilon = 1e-5;         % Regularization
sigma   = cov(X, 1);    % As per ANG's notes, covariance is computed as X*X'/N, 
                        % however cov(X) computes X*X'/(N-1), known as Bessel's correction 
                        % and apparently is the correct way of computing covariance. 
                        % cov(X,1) computes the same as X*X'/N
[U S V] = svd(sigma);   % Perform svd on cov matrix
X       = X*U;                                  % Rotated (considering column vectors, u'x is performed but since x is row and u is column, this works just the same
X       = X*diag(1./sqrt(diag(S) + epsilon));   % normalize as in PCA whitening (refer UFLDL exercise)

end