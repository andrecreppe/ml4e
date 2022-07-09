clear; close all; clc
%%% Exercise 2: Gaussian Process Regression

%% 2a)
% Plot the Kernel Function for fixed x1 = 0 and varying x2 in {-5,5} for
% varying theta_f and theta_l
x1 = 0;
x2 = linspace(-5, 5);

% Use a (nested) for-loop to plot the kernel for different combinations of
% the kernel parameters
for thetaF = [0.5, 2]
    for thetaL = [0.5, 2]
        k = SqExpKernel(x1, x2, thetaF, thetaL);
        
        %figure
        %plot(k)
        %title("tF=" + thetaF + "; tL=" + thetaL)
    end
end

%% 2b)
% Now we want to sample from the prior distribution over functions of the Gaussian process: The multivariate
% gausian distribution is defined by its mean and covariance. The mean is
% assumed to be constant 0 and the covariance matrix is defined by the
% kernel function.

% Construct the covariance matrix with a nested for loop.
N = 100;
X = linspace(0, 10, N);
samp = 20;
mu = zeros(N, 1);

for i=1:N
    for j=1:N
        C(i,j) = SqExpKernel(X(i), X(j), 0.5, 0.5);
    end
end

% Use the mvnrnd() function to sample from the multivariate Gaussian
% distribution
Z = mvnrnd(mu, C, samp);
    
% Plot the sampled functions
plot(X, Z)
    
%% 2c)
% Conditional distribution: The joint distribution of our training data y
% and the output we want to predict f_* is, as we know, also multivariate
% Gaussian distributed with the covariance matrix C. From the joint distribution we can 
% derive the conditional distribution to make new predictions f_* given query inputs x_*.
% For this exercise use the function 'GenerateNonlinData' to generate training data.
% We assume additive gaussian noise in the output.

% Generate non linear data
...

% Make predictions using your implementation of the gprPred() function
...

% Plot the results
...

%% Function to make predictions using GPR with the squared exponential kernel
function [mu_pred,sig_pred] = gprPred(x,y,xNew,thetaF,thetaL,noiseP)
    N = length(x);
    covn = zeros(N);
    
    % Construct a (nested) for-loop to construct the covariance matrix K_y
    for i=1:N
        for j=1:N
            C(i,j) = SqExpKernel(x(i), x(j), thetaF, thetaL);
        end
    end
    
    % Add noise to the covariance matrix K_y
    conv = conv + noiseP * eye(N);
    
    % Create k_* with the new observation x_*
    ...
    
    % Make a prediction for the mean and the variance
    ...
    
end


%% Generate nonlinear Data
function [x,y] = GenerateNonlinData(N,noise)

% Nonlinear test function (http://www.sfu.ca/~ssurjano/forretal08.html)

x = linspace(0,1,N)';
y = ((6*x-2).^2).*sin(12*x-4) + noise*randn(N,1);

end

%% Implementation of the squared exponential Kernel Function
function k = SqExpKernel(x1,x2,thetaF,thetaL)
    k = (thetaF.^2) * exp(-((x1-x2).^2)/(2.*(thetaL.^2)));
end