clear; close all; clc

%% 2d)
% Application to real world problem using the carsmall data set and MATLAB ML-Toolbox

% Load the carsmall data set with 'Displacement','Horsepower','Weight' as predictors

...

%% i)
% Fit a GPR to the data

...
    
% Predict the output for the training data using the predict() function
...
    
% Plot the predicted vs. the training data
...

% Calculate MSE
...
    
%% ii)
% Predict MPG for variying 'Horsepower' with constant remaining predictors
...

% Plot the prediction together with the 95 % confidence interval
...

%% iii)
% Repeat i) and ii) with different kernel functions
...
    
%% iv)
% Estimate a linear regression model with the linmod function and compare the results with your GPR model
