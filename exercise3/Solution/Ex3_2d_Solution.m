%% Exercise 3-2, d): Gaussian Process Regression
% Felix Wittich, 28.06.2022
%%
clear; close all; clc

%% 2di)
% Application to real world problem using the carsmall data set and MATLAB ML-Toolbox

% Load the carsmall data set with 'Displacement','Horsepower','Weight' as predictors

load carsmall
ds = table(Displacement,Horsepower,Weight,MPG, ... 
    'VariableNames',{'Displacement','Horsepower','Weight','MPG'});

% Fit a GPR to the data
gprMdl = fitrgp(ds,'MPG','KernelFunction','ardsquaredexponential');

% Predict the output for the training data
[ypred,ysd,yint] = predict(gprMdl,[ds(:,1) ds(:,2) ds(:,3)]);

% Plot prediction vs. the training data
figure
hold on
plot(ds.MPG,ypred,'x')
plot([min(ds.MPG) max(ds.MPG)],[min(ds.MPG) max(ds.MPG)])
xlabel('Measured values')
ylabel('Predicted values')

% Calculate MSE
MSE = nanmean((ypred-ds.MPG).^2);

%% 2dii)
% Predict MPG for varying 'Horsepower' with constant remaining predictors
hpnew = linspace(0,300,100)';
xnew = [repmat(200,100,1) hpnew repmat(3000,100,1)];

[ynew,ysdnew,yintnew] = predict(gprMdl,xnew);

% Plot the prediction together with the 95 % confidence interval
figure
hold on
plot(hpnew,ynew,'b')
plot(hpnew,yintnew(:,1),'--r')
plot(hpnew,yintnew(:,2),'--r')
legend('Prediction','95 % Confidence Interval')
xlabel('Horsepower')
ylabel('MPG')
axis([0 300 0 45])

%% 2diii)
% Repeat i) and ii) with different kernel functions

%% 2div)
% Estimate a linear regression model with the linmod function and compare the results with your GPR model

linmod = fitlm(ds); % Fit a linear regression model

rmse_gpr = sqrt(MSE); % Calculate the RMSE for the GPR model
rmse_lin = linmod.RMSE; % Calculate the RMSE for the linear model

figure % Plot predictions for GPR and linear model
hold on
plot(hpnew,ynew,'b')
plot(hpnew,predict(linmod,xnew),'r')
plot(Horsepower,MPG,'x')
xlabel('Horsepower')
ylabel('MPG')
legend('GPR','Linear')