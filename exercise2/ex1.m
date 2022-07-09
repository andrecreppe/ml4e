%% INITIALIZATION
close all; clear all; clc
rng(1000);

%% 1a
N = 1000;
X = randn(N, 1);

sd = 0.8;
e = sd*randn(N, 1);

%% 1b
b0 = 2;  b1 = 3;
b2 = -1; b3 = 2;

Y = b0 + b1*X + b2*(X.^2) + b3*(X.^3) + e;

%% 1c
Xtr = X(1:100); Xts = X(101:end);
Ytr = Y(1:100); Yts = Y(101:end);

figure;
plot(Xtr, Ytr, '.');
title('Training Data')

%% 1d
for i = 1:9
    PhiTr(:, i) = Xtr.^i;
    PhiTs(:, i) = Xts.^i;
    
    model = fitlm(PhiTr, Ytr);
    AIC(i) = model.ModelCriterion.AIC;
    BIC(i) = model.ModelCriterion.BIC;
    R2(i) = model.Rsquared.Ordinary;
    RMSE(i) = model.RMSE;
    
    YhatTr = predict(model, PhiTr);
    YhatTs = predict(model, PhiTs);
    
    RMSEts(i) = sqrt(mean((Yts - YhatTs).^2));
end

%% 1e
figure
subplot 221; plot(AIC);  title('AIC');
subplot 222; plot(BIC);  title('BIC');
subplot 223; plot(R2);   title('R2');
subplot 224; plot(RMSE); title('RMSE');

%% 1f

% ---------- terminar