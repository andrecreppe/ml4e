%% Exercise 2-1 - Getting Familiar with Stepwise Model Selection in Matlab
% Felix Wittich, 01.06.2021
%%
clear all
close all
clc

%% a)
% Use the randn() function to generate a predictor Xtrain of length
% N = 1000, as well as a noise vector eps of length N = 1000 with a
% standard deviation of 0.8. Make sure to use rng(1000) prior to
% staring part a) to ensure consistent results.

rng(1000)

N = 1000;
X = randn(N,1);
eps = 0.8*randn(N,1);


%% b)
% Generate a response vector Y of length n = 100 according to
% the model
% Y = beta_0 + beta_1X + beta_2X^2 + beta_3X^3 + epsilon,
% where beta_0=2, beta_1=3, beta_2=-1, and beta_3=0.5. Set seed to 1998.

beta_0 = 2;
beta_1 = 3;
beta_2 = -1;
beta_3 = 2;
Y = beta_0 + beta_1*X + beta_2*X.^2 + beta_3*X.^3 + eps;

%% c)
Ytest = Y(101:end);
Ytrain = Y(1:100);
Xtest = X(101:end,:);
Xtrain = X(1:100,:);

%% d)
% Use a for loop to generate a set of increasing nested models containing the
% predictors X, X^2,..., X^9. For each model, determine the AIC, BIC, and
% R2 and plot the results. What is the best model according to these
% criteria?
figure
plot(Xtrain,Ytrain,'x')
xlabel('X')
ylabel('Y')
hold all
for li = 1:9
    PhiTrain(:,li) = Xtrain.^li;
    PhiTest(:,li) = Xtest.^li;
    myModel = fitlm(PhiTrain,Ytrain);
    AIC(li) = myModel.ModelCriterion.AIC;
    BIC(li) = myModel.ModelCriterion.BIC;
    R2(li) = myModel.Rsquared.Ordinary;
    RMSE(li) = myModel.RMSE;
    
    YhatTrain = predict(myModel,PhiTrain);
    YhatTest = predict(myModel,PhiTest);
    
    RMSEtest(li) = sqrt(mean((Ytest-YhatTest).^2));
    
%     [~,idx] = sort(Xtrain);
%     plot(Xtrain(idx),YhatTrain(idx))
    
end

%% e)
figure
subplot 221
plot(AIC)
xlabel('nb. of regressors')
ylabel('AIC')
subplot 222
plot(BIC)
xlabel('nb. of regressors')
ylabel('BIC')
subplot 223
plot(R2)
xlabel('nb. of regressors')
ylabel('R2')
subplot 224
plot(RMSE)
hold on
plot(RMSEtest)
xlabel('nb. of regressors')
ylabel('RMSE')
legend('training','test')
%% f)
% Now use the stepwiselm() function to perform a stepwise selection
% in order to choose the best model containing the predictors
% X,X2, . . .,X9. What is the best model obtained according to
% AIC, BIC, and R2? 

myModel1 = stepwiselm(Xtrain,Ytrain,'poly9','Criterion','AIC')
myModel2 = stepwiselm(Xtrain,Ytrain,'poly9','Criterion','BIC')
myModel3 = stepwiselm(Xtrain,Ytrain,'poly9','Criterion','Rsquared')

% figure
% plot(Ytest,predict(myModel1,Xtest),'+')
% figure
% plot(Ytest,predict(myModel2,Xtest),'+')
% figure
% plot(Ytest,predict(myModel3,Xtest),'+')