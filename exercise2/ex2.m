%% INITIALIZATION
close all; clear all; clc
rng(1000);

%% 2a
N = 1000; X = randn(N, 1);
sd = 0.8; e = sd*randn(N, 1);

b0 = 2;  b1 = 3;
b2 = -1; b3 = 2;
Y = b0 + b1*X + b2*(X.^2) + b3*(X.^3) + e;

Xtr = X(1:100); Xts = X(101:end);
Ytr = Y(1:100); Yts = Y(101:end);

%figure;
%plot(Xtr, Ytr, '.');
%title('Training Data')

%% 2b
lambda = logspace(2, -5, 100);

%% 2c
for i = 1:9
    PhiTr(:, i) = Xtr.^i;
    PhiTs(:, i) = Xts.^i;
end

[Beta, FitInfo] = lasso(PhiTr, Ytr, 'Lambda', lambda, 'CV', 10, 'PredictorNames', {'x1','x2','x3','x4','x5','x6','x7','x8','x9'});

%% 2d
lassoPlot(Beta, FitInfo, 'PlotType', 'CV', 'PredictorNames', {'x1','x2','x3','x4','x5','x6','x7','x8','x9'});
lassoPlot(Beta, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log', 'PredictorNames', {'x1','x2','x3','x4','x5','x6','x7','x8','x9'});

beta0 = FitInfo.Intercept(FitInfo.IndexMinMSE);
betaLasso = [beta0 Beta(:, FitInfo.IndexMinMSE)];

%% 2e

% ---------- terminar
