%% Machine Learning Online Class
%  Exercise 6 | Support Vector Machines
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% =============== Part 1: Loading and Visualizing Data ================
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data1: 
% You will have X, y in your environment
load('ex6data1.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Training Linear SVM ====================
%  The following code will train a linear SVM on the dataset and plot the
%  decision boundary learned.
%

% Load from ex6data1: 
% You will have X, y in your environment
load('ex6data1.mat');

fprintf('\nTraining Linear SVM ...\n')

% You should try to change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)
% cv = [16384 8192 4096 2048 1024 512 256 128 100 64 32 16 8 4 2 1 0.3 0.1 0.03 0.01 0.003 0.001 0];
% cv = [1 3 10 30 100 300 1000 3000 10000 30000 100000 300000 1000000];
cv = [100];
while (length(cv))
C = cv(1);
cv = cv (2:length(cv));
a = 1/C;
fprintf('C = %f, a = %f..... ', C, a);
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
% hold on;
visualizeBoundaryLinear(X, y, model);
fprintf('C = %f, a = %f. %d iteration(s) to go\n', C, a, length(cv));
pause;
end
hold off;

%% =============== Part 3: Implementing Gaussian Kernel ===============
%  You will now implement the Gaussian kernel to use
%  with the SVM. You should complete the code in gaussianKernel.m
%
fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :' ...
         '\n\t%f\n(this value should be about 0.324652)\n'], sim);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 4: Visualizing Dataset 2 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data2.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
%  After you have implemented the kernel, we can now use it to train the 
%  SVM classifier.
% 
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 6: Visualizing Dataset 3 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

Cs = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmas = [0.01 0.03 0.1 0.3 1 3 10 30];
inputs = zeros(length(Cs) * length(sigmas), 2); 
scores = zeros(length(inputs), 1);

i = 1;
for C = Cs
    for sigma = sigmas
      inputs(i,1) = C;
      inputs(i,2) = sigma;
      fprintf('C = %f, sigma = %f\n', C, sigma);
      i = i+1;
    end
end

i = 1;
for input = inputs'
    C = input(1);
    sigma = input (2);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model,Xval);
    score = mean(double(predictions ~= yval));
    scores(i, 1) = score;
    fprintf('score = %f, C = %f, sigma = %f\n', score, C, sigma);
    i = i+1;
end
    [winner,winnerIdx] = min(scores);
    input = inputs(winnerIdx,:);
    C = input(1);
    sigma = input (2);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    visualizeBoundary(X, y, model);
    fprintf('winner: C = %f, sigma = %f. score = %f, idx = %d \n', C, sigma, winner, winnerIdx);
    pause;

% Train the SVM
% fprintf('C = %f, sigma = %f\n', C, sigma);
% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
% visualizeBoundary(X, y, model);
% score = svmPredict(model,X);
% fprintf('score = %d, C = %f, sigma = %f\n', score, C, sigma);
% pause;

% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
% visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

