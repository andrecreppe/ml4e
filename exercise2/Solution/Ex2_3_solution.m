%% Exercise 2-3 - Model Selection for mpg Data Set
% Felix Wittich, 01.06.2021
%%

clear all
close all
clc
rng(1)

%% a)
%Load the carsmall data set and choose the displacement, weight, horsepower, and
%acceleration as potential input variables to predict mpg. Store all variables in a
%matrix and use data(any(isnan(data),2),:) = [] to get rid of NaN values. Plot
%the correlation between predictors and output and determine the corresponding
%correlation coefficients using corrcoef(). What can be concluded from the results?

load carsmall

data = [Displacement,Weight,Horsepower,Acceleration,MPG];
data(any(isnan(data),2),:) = [];

X = data(:,1:end-1);
Y = data(:,end);

figure
% plot correlation
[~,ax] = plotmatrix([Y X]);
% set labels (optional)
ylabel(ax(1,1),'MPG')
ylabel(ax(2,1),'Disp')
ylabel(ax(3,1),'Weight')
ylabel(ax(4,1),'Horse')
ylabel(ax(5,1),'Accel')
xlabel(ax(5,1),'MPG')
xlabel(ax(5,2),'Disp')
xlabel(ax(5,3),'Weight')
xlabel(ax(5,4),'Horse')
xlabel(ax(5,5),'Accel')
% determine correlation coefficients
R = corrcoef([Y X])

%% b)
% choose model type
type = 'linear';

% perform stepwise selection using different criteria
myModelAIC = stepwiselm(X,Y,type,'Upper','linear','Criterion','AIC')
myModelBIC = stepwiselm(X,Y,type,'Upper','linear','Criterion','BIC')
myModelRsquared = stepwiselm(X,Y,type,'Upper','linear','Criterion','Rsquared')

% use lasso for selection

Phi = x2fx(X,'linear'); % generate regression matrix

[B, Stats] = lasso(Phi(:,2:end),Y,'CV',10); % estimate using lasso

lassoPlot(B, Stats, 'PlotType', 'CV') % plot CV results
lassoPlot(B, Stats, 'PlotType', 'Lambda','XScale','log') % plot coefficient path
ylabel('value of beta')


beta_0_Lasso = Stats.Intercept(Stats.IndexMinMSE);
BetaLasso = [beta_0_Lasso B(:,Stats.IndexMinMSE)']';

% re-estimate model selected by lasso in order to obtain unbiased estimate
myModelLasso = fitlm(X(:,BetaLasso(2:end)~=0),Y)

% estimate full model
myModel = fitlm(X,Y)

% evaluate models in CV
yFit = @(XTrain,yTrain,XTest)(XTest*regress(yTrain,XTrain));

Xtest = [ones(length(Y),1) X];
Ytest = Y;

cvMSEmyModelAIC = crossval('MSE',...
    Xtest(:,[true; myModelAIC.VariableInfo.InModel(1:end-1)]),...
    Ytest,'predfun',yFit);
cvRMSEmyModelAIC = sqrt(cvMSEmyModelAIC);

cvMSEmyModelBIC = crossval('MSE',...
    Xtest(:,[true; myModelBIC.VariableInfo.InModel(1:end-1)]),...
    Ytest,'predfun',yFit);
cvRMSEmyModelBIC = sqrt(cvMSEmyModelBIC);

cvMSEmyModelRsquared = crossval('MSE',...
    Xtest(:,[true; myModelRsquared.VariableInfo.InModel(1:end-1)]),...
    Ytest,'predfun',yFit);
cvRMSEmyModelRsquared = sqrt(cvMSEmyModelRsquared);

cvMSEmyModelLasso = crossval('MSE',...
    Xtest(:,BetaLasso~=0),Ytest,'predfun',yFit);
cvRMSEmyModelLasso = sqrt(cvMSEmyModelLasso);

cvMSEmyModel = crossval('MSE',...
    Xtest,Ytest,'predfun',yFit);
cvRMSEmyModel = sqrt(cvMSEmyModel);

% compare models
RowNames = {'myModelfull','myModelAIC','myModelBIC',...
    'myModelRsquared','myModelLasso'};
RMSE = [myModel.RMSE;myModelAIC.RMSE;myModelBIC.RMSE;...
    myModelRsquared.RMSE;myModelLasso.RMSE];
AIC = [myModel.ModelCriterion.AIC;...
    myModelAIC.ModelCriterion.AIC;myModelBIC.ModelCriterion.AIC;...
    myModelRsquared.ModelCriterion.AIC;myModelLasso.ModelCriterion.AIC];
BIC = [myModel.ModelCriterion.BIC;...
    myModelAIC.ModelCriterion.BIC;myModelBIC.ModelCriterion.BIC;...
    myModelRsquared.ModelCriterion.BIC;myModelLasso.ModelCriterion.BIC];
cvRMSE = [cvRMSEmyModel;cvRMSEmyModelAIC;cvRMSEmyModelBIC;...
    cvRMSEmyModelRsquared;cvRMSEmyModelLasso];
dimTheta = [1+size(X,2);...
    1+sum(myModelAIC.VariableInfo.InModel(1:end-1));...
    1+sum(myModelBIC.VariableInfo.InModel(1:end-1));...
    1+sum(myModelRsquared.VariableInfo.InModel(1:end-1));...
    1+sum(BetaLasso(2:end)~=0)];
Models = table(RMSE,AIC,BIC,cvRMSE,dimTheta,...              
               'RowNames',RowNames)

%% c)
clc, close all

X = data(:,3);

% choose model type
type = 'poly9';

% perform stepwise selection using differenkt criteria
myModelAIC = stepwiselm(X,Y,type,'Criterion','AIC')
myModelBIC = stepwiselm(X,Y,type,'Criterion','BIC')
myModelRsquared = stepwiselm(X,Y,type,'Criterion','Rsquared')

% use lasso for selection
for li = 1:9
    PhiPoly(:,li) = X.^li; % generate regression matrix
end

X = PhiPoly;

[B, Stats] = lasso(X,Y,'CV',10); % estimate using lasso

lassoPlot(B, Stats, 'PlotType', 'CV') % plot CV results
lassoPlot(B, Stats, 'PlotType', 'Lambda','XScale','log',...
    'PredictorNames',{'x1','x2','x3','x4','x5','x6','x7','x8','x9'}) % plot coefficient path
ylabel('value of beta')

beta_0_Lasso = Stats.Intercept(Stats.IndexMinMSE);
BetaLasso = [beta_0_Lasso B(:,Stats.IndexMinMSE)']';

% re-estimate model selected by lasso in order to obtain unbiased estimate
myModelLasso = fitlm(X(:,BetaLasso(2:end)~=0),Y);

% estimate full model
myModel = fitlm(X,Y)

% estimate linear model
myModelLinear = fitlm(X(:,1),Y)

% evaluate models in CV
yFit = @(XTrain,yTrain,XTest)(XTest*regress(yTrain,XTrain));

Xtest = [ones(length(Y),1) X];
Ytest = Y;

inModelAIC = false(9,1);
inModelAIC(myModelAIC.Formula.Terms(2:end,1)) = true;
inModelBIC = false(9,1);
inModelBIC(myModelBIC.Formula.Terms(2:end,1)) = true;
inModelRsquared = false(9,1);
inModelRsquared(myModelRsquared.Formula.Terms(2:end,1)) = true;

cvMSEmyModelAIC = crossval('MSE',...
    Xtest(:,[true; inModelAIC]),...
    Ytest,'predfun',yFit);
cvRMSEmyModelAIC = sqrt(cvMSEmyModelAIC);

cvMSEmyModelBIC = crossval('MSE',...
    Xtest(:,[true; inModelBIC]),...
    Ytest,'predfun',yFit);
cvRMSEmyModelBIC = sqrt(cvMSEmyModelBIC);

cvMSEmyModelRsquared = crossval('MSE',...
    Xtest(:,[true; inModelRsquared]),...
    Ytest,'predfun',yFit);
cvRMSEmyModelRsquared = sqrt(cvMSEmyModelRsquared);

cvMSEmyModelLasso = crossval('MSE',...
    Xtest(:,BetaLasso~=0),Ytest,'predfun',yFit);
cvRMSEmyModelLasso = sqrt(cvMSEmyModelLasso);

cvMSEmyModel = crossval('MSE',...
    Xtest,Ytest,'predfun',yFit);
cvRMSEmyModel = sqrt(cvMSEmyModel);

cvMSEmyModelLinear = crossval('MSE',...
    Xtest(:,[1,2]),Ytest,'predfun',yFit);
cvRMSEmyModelLinear = sqrt(cvMSEmyModelLinear);

% compare models
RowNames = {'myModelLinear','myModelfull','myModelAIC','myModelBIC',...
    'myModelRsquared','myModelLasso'};
RMSE = [myModelLinear.RMSE;myModel.RMSE;myModelAIC.RMSE;myModelBIC.RMSE;...
    myModelRsquared.RMSE;myModelLasso.RMSE];
AIC = [myModelLinear.ModelCriterion.AIC;myModel.ModelCriterion.AIC;...
    myModelAIC.ModelCriterion.AIC;myModelBIC.ModelCriterion.AIC;...
    myModelRsquared.ModelCriterion.AIC;myModelLasso.ModelCriterion.AIC];
BIC = [myModelLinear.ModelCriterion.BIC;myModel.ModelCriterion.BIC;...
    myModelAIC.ModelCriterion.BIC;myModelBIC.ModelCriterion.BIC;...
    myModelRsquared.ModelCriterion.BIC;myModelLasso.ModelCriterion.BIC];
cvRMSE = [cvRMSEmyModelLinear;cvRMSEmyModel;cvRMSEmyModelAIC;...
    cvRMSEmyModelBIC;cvRMSEmyModelRsquared;cvRMSEmyModelLasso];
dimTheta = [2;1+size(X,2);...
    1+sum(inModelAIC);...
    1+sum(inModelBIC);...
    1+sum(inModelRsquared);...
    1+sum(BetaLasso(2:end)~=0)];
Models = table(RMSE,AIC,BIC,cvRMSE,dimTheta,...              
               'RowNames',RowNames)
