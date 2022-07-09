clear; close all; clc;

%% Exercise 1
% 1ab) kNN regression with different k

rng(7); % Fix random number generator for reproducible results

% Generate data using the generate_nonlin_data_1D() function. And make
% predictions using your implementation of the kNN algorithm
N = 50; 
nl = 1; % noise level
k = 1;

[dataX, dataY] = generate_nonlin_data_1D(N,nl);

for i=1:N
    pred(i) = knn_regression(dataX, dataY, dataX(i), k);
end

% Plot results
figure
plot(dataX, dataY, 'x');
hold on
plot(dataX, pred, 'r');

% Implementation of the basic kNN algorithm
function y_pred = knn_regression(x_train, y_train, x0, k)
    x = x_train;
    y = y_train;

    % Calculate pairwise euclidean distance
    dist = sqrt(sum((x - x0).^2)); 

    % Sort the distances in an ascendig order (use the in-built function 'sort')
    [sorted, ind] = sort(dist);

    % Find the k closest output values
    ynn = y(ind(1:k), :);

    % Calculate the mean of the k closest output values
    y_pred = (1/k) * sum(ynn);
end

% Data generating function
function [x,y] = generate_nonlin_data_1D(N, nl)
    % 1D Nonlinear test function (http://www.sfu.ca/~ssurjano/forretal08.html)
    x = linspace(0,1,N)';
    y = ((6*x-2).^2).*sin(12*x-4) + nl*randn(N,1);
end