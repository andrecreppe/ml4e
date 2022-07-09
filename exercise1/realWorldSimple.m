% Exercise 1.2 - Simple Linear Regression: Real World Problem

%Initialization
close all
clear all
clc

%2a
load carsmall
ds = table(Horsepower, MPG, 'VariableNames', {'Horsepower', 'MPG'});

linMod = fitlm(ds);
disp(linMod);

precision = 98;
[pred, predCI] = linMod.predict(precision);
[~, predPI] = linMod.predict(precision, 'Prediction', 'Observation');

%2b
figure
plot(linMod);

%2c
figure
subplot(1,3,1) %if it is normaly distribuited
plotResiduals(linMod,'probability')
subplot(1,3,2) %if the data is linear or not (linearity)
plotResiduals(linMod, 'fitted')
subplot(1,3,3) %which datapoints are outliers
plotDiagnostics(linMod)
legend('show')
