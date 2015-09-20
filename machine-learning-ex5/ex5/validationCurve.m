function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
% lambda_vec = [0 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12 10.24]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

    for i = 1:length(lambda_vec)
        lambda = lambda_vec(i);
        % Compute train / val errors when training linear 
        % regression with regularization parameter lambda
        theta = trainLinearReg(X, y, lambda);
        [errorTrainI,gradTrainI]=linearRegCostFunction(X,y,theta,0);
        [errorValI,gradValI]=linearRegCostFunction(Xval,yval,theta,0);
        % store the result in error_train(i) and error_val(i)
        error_train(i) = errorTrainI;
        error_val(i) = errorValI;
    end








% =========================================================================

end