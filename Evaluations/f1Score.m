function [f1, precision, recall] = f1Score(predicted, actual)

%% function determines the f1 score of an algorithm given 
%  predicted values and actual values

% Note: Currently accepts only binary classification (0,1). probabilistic
% values of predicted and actual will be rounded to 0 or 1

% Future mods:
% 1. include options for specificity and sensitivity

predicted = round(predicted);
actual    = round(actual);

truePosMat  =  predicted &  actual;     % True positive if predicted & actual
falsePosMat =  predicted & ~actual;     % False positive if predicted but not actual
falseNegMat = ~predicted &  actual;     % False negative if not predicted when actual
trueNegMat  = ~predicted & ~actual;     % True negative if not predicted and not actual

tp = sum(truePosMat(:));
fp = sum(falsePosMat(:));
fn = sum(falseNegMat(:));
tn = sum(trueNegMat(:));

precision = tp./(tp+fp);
recall    = tp./(tp+fn);

f1 = 2*precision*recall/(precision+recall);

end