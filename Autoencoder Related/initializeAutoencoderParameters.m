function theta = initializeAutoencoderParameters(visibleSize, hiddenSize)

%% Initialize parameters randomly based on layer sizes.

% Credits:
% 1. Adaped from UFLDL Tutorial
% (http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)

r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(visibleSize, hiddenSize) * 2 * r - r;
W2 = rand(hiddenSize, visibleSize) * 2 * r - r;

b1 = zeros(1, hiddenSize);
b2 = zeros(1, visibleSize);

% Convert weights and bias gradients to the vector form.
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

