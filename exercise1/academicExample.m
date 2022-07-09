% Exercise 1.1 - Simple Linear Regression: Academic Example

%Initialization
close all
clear all
clc

rng(1000);

%1a
N = 50;
X = linspace(1, 100, N);
X = X';

%1b
sigma = 30;
e = 30*randn(N, 1);

%1c
beta0 = 50;
beta1 = 7;
Y0 = beta0 + beta1*X;
Y = Y0 + e;

%1d
figure;
scatter(X, Y, ".");

%1e
Phi = [ones(size(X)) X]; %Y = b0 + b1X -> phi = [b0, b1] // theta = [1, X]
theta = Phi\Y

%1f
Yh = Phi*theta;

hold all
plot(X, Yh)
plot(X, Y0)
legend('Data', 'LS estimate', 'Y0')

%1g
model1 = fitlm(X, Y) %information extra than the backslash
figure
plot(model1)

%1h
model2 = fitlm(X, Y, 'poly2')
figure
plot(model2)

%1i and 1j
% just a for loop to see how sample size and deviation change the model
