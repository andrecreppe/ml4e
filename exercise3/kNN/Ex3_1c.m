clear; close all; clc;

%% Exercise 3
% 1c) Leave-One-Out Cross-Validation with kNN regression

rng(5); % Fix random number generator for reproducible results

% Generate non-linear data
...

% For-loop to implement the LOOCV for the MSE with kNN regression of the
% generated data set
...

% Plot MSE_LOOCV for different k
...
    
% Implementation of the basic kNN algorithm
function y_pred = knn_regression(x_train,y_train,x0,k)

x = x_train;
y = y_train;

% Calculate pairwise euclidean distance
dist = sum(sqrt((x - x0).^2),2); 
% dist = pdist([repmat(x0,size(x,1),1) x],'euclidean');

% Sort the distances in an ascendig order (use the in-built function 'sort')
[sdist,ind] = sort(dist);

% Find the k closest output values
ynn = y(ind(1:k),:); 

% Calculate the mean of the k closest output values
y_pred = (1/k)*(sum(ynn)); 

end

% Data generating function
function [x,y] = generate_nonlin_data_1D(N,nl)

% 1D Nonlinear test function (http://www.sfu.ca/~ssurjano/forretal08.html)

x = linspace(0,1,N)';
y = ((6*x-2).^2).*sin(12*x-4) + nl*randn(N,1);

end