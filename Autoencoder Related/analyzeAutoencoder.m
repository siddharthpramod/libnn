function analyzeAutoencoder(theta, data, inputSize, hiddenSize)

%% Script to analyze performance of an autoencoder.
%  Produces a plot of reconstruction error and input v/s reconstruction

% Future mods:
% 1. Add options

[W1, W2, b1, b2] = unrollAutoencoderTheta(theta, inputSize, hiddenSize);
z2 = bsxfun(@plus, data*W1, b1);    % m x hidden
a2 = sigmoid(z2);
z3 = bsxfun(@plus, a2*W2, b2);      % m x visible
a3 = sigmoid(z3);

fprintf('\n Mean Absolute error is ');
meanError = mean(abs(a3(:)-data(:)))

fprintf('\n Mean Autoencoder activation is ');
meanActiv = mean(a2(:))

figure(1);
plot(a3(:) - data(:));
 
% figure(2);
% scatter(a3(:), data(:), '.');
% 
figure(3);
hist(data(:), 50);

figure(4);
hist(a3(:), 50);

[f1, precision ,recall] = f1Score(a3,data)
%
% for i = (1:size(a3,2))
%     figure(7);
%     hold;
%     plot(a3(i,:));
%     plot(data(i,:));
%     hold off;
%     pause;
% end
% 
% figure(8);
% plot(a3(:,6));

end