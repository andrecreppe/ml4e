%% Simple Linear Regression
% Farzad Rezazadeh Pilehdarboni, 25.05.2022

close all;clear;clc
%% 2a)
% Load the data set
load carsmall
ds = table(Horsepower,MPG,'VariableNames',{'Horsepower','MPG'});

% Fit the simple lineare model
lin_mod = fitlm(ds);

% Display the results of the estimation
disp(lin_mod)

% Prediction for Horsepower = 98 with confidence and prediction interval
[pred,predCI] = lin_mod.predict(98);
[~,predPI] = lin_mod.predict(98,'Prediction','observation');

%% 2b)

%Plot the training data and the regression model with confidence bounds
figure(1)
plot(lin_mod)

%% 2c)

% Diagnostic plots
figure(2)
plotResiduals(lin_mod,'probability')
legend('show')
figure(3)
plotResiduals(lin_mod,'fitted')
legend('show')
figure(4)
plotDiagnostics(lin_mod)
legend('show')