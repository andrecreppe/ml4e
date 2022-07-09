%% INITIALIZATION
close all; clear all; clc
rng(1)

%% 3a

load carsmall % Displacement, Weight, Horsepower, Acceleration >> MPG
clear Cylinders Model Mfg Origin Model_Year

data = [Displacement, Weight, Horsepower, Acceleration, MPG];
data(any(isnan(data), 2), :) = [];

X = data(:, 1:end-1);
Y = 

% ---------- terminar
