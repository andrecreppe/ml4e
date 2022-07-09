%% Multiple Linear Regression
% Farzad Rezazadeh Pilehdarboni, 25.05.2022
close all;clear;clc

%% 3a)

% Load the data set
load carsmall
ds = table(Acceleration,Cylinders,Displacement,Horsepower,Model_Year,Weight,MPG, ... 
    'VariableNames',{'Acceleration','Cylinders','Displacement','Horsepower','Model_Year','Weight','MPG'});

% Scatterplot matrix of data set
figure(1)
corrplot(ds(:,1:7))

%% 3b)

% Fit multiple regression model with all predictors
mult_mod = fitlm(ds);
disp(mult_mod)

% Fit multiple regression model with only Model_Year and Weight
mult_mod_2 = fitlm(ds,'MPG ~ 1 + Model_Year + Weight');
disp(mult_mod_2)

%% 3c)

% Diagnostic plots
figure(2)
plotResiduals(mult_mod,'probability')
legend('show')
figure(3)
plotResiduals(mult_mod,'fitted')
legend('show')
figure(4)
plotDiagnostics(mult_mod)
legend('show')

%% 3d)

% Add interaction terms
mult_mod_inter = fitlm(ds,'interactions');
disp(mult_mod_inter)

%% 3e)

% Add quadratic terms
transf_mult_mod = fitlm(ds, ...
    'MPG ~ 1 + Acceleration + Horsepower + Horsepower^2 + Weight + Displacement + Displacement^2 + Cylinders + Cylinders^2 + Model_Year');
disp(transf_mult_mod)

