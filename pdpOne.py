__author__ = 'maruthi'

import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import random
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.datasets.california_housing import fetch_california_housing

colNamesTP = ['Stars', 'Food_Avg', 'Service_Avg', 'Price_Avg', 'Ambience_Avg']
gTpdR = pd.read_csv('dataset/aspectsPdpOne.csv',skiprows=0, sep=',',names=colNamesTP)
pdpAZ = pd.read_csv('dataset/aspectsPdpAZ.csv',skiprows=0, sep=',',names=colNamesTP)
pdpNV = pd.read_csv('dataset/aspectsPdpNV.csv',skiprows=0, sep=',',names=colNamesTP)

# Float coversion from 64 to 32
# gTpdR = np.zeros(gTpdR.shape[1], dtype = np.float32) # to convert float64 to float32

# All
predictoR1 = gTpdR[gTpdR.columns - ['Stars']]
targeT1 = gTpdR['Stars']
dataRS1 = random.sample(gTpdR.index, int(len(gTpdR)*.80))
predictorTrain1, targetTrain1 = predictoR1.ix[dataRS1],targeT1.ix[dataRS1]
predictorTest1, targetTest1  = predictoR1.drop(dataRS1),targeT1.drop(dataRS1)

dataGBR1 = GradientBoostingRegressor(n_estimators=1000, max_depth=4,
                                learning_rate=0.1, loss='huber',
                                random_state=1).fit(predictorTrain1, targetTrain1)


msE1 = mean_squared_error(targetTest1, dataGBR1.predict(predictorTest1))
R21 = r2_score(targetTest1, dataGBR1.predict(predictorTest1))

print("All MSE: %.4f" % msE1)
print("All R2: %.4f" % R21)


pltNameData = ['Food_Avg', 'Service_Avg', 'Price_Avg', 'Ambience_Avg']
pltData = [0, 1, 2, 3] # Which columns to take input from the predictors train dataset
fig1, axs = plot_partial_dependence(dataGBR1, predictorTrain1, pltData, feature_names=pltNameData,n_cols=2, grid_resolution=50)

fig1.suptitle('Aspect Sentiments PDP\n'
             'Partial Dependency of users review sentiments against their star ratings')
plt.subplots_adjust(top=0.9)
fig1 = plt.figure()
plt.show()


# AZ

predictoR2 = pdpAZ[pdpAZ.columns - ['Stars']]
targeT2 = pdpAZ['Stars']
dataRS2 = random.sample(pdpAZ.index, int(len(pdpAZ)*.80))
predictorTrain2, targetTrain2 = predictoR2.ix[dataRS2],targeT2.ix[dataRS2]
predictorTest2, targetTest2  = predictoR2.drop(dataRS2),targeT2.drop(dataRS2)

dataGBR2 = GradientBoostingRegressor(n_estimators=100, max_depth=10":"+,
                                learning_rate=0.1, loss='huber',
                                random_state=1).fit(predictorTrain2, targetTrain2)


msE2 = mean_squared_error(targetTest2, dataGBR2.predict(predictorTest2))
R22 = r2_score(targetTest2, dataGBR2.predict(predictorTest2))

print("All MSE: %.4f" % msE2)
print("All R2: %.4f" % R22)


pltNameData = ['Food_Avg', 'Service_Avg', 'Price_Avg', 'Ambience_Avg']
pltData = [0, 1, 2, 3] # Which columns to take input from the predictors train dataset
fig2, axs = plot_partial_dependence(dataGBR2, predictorTrain2, pltData, feature_names=pltNameData, n_jobs=1, n_cols=2, grid_resolution=50)

fig2.suptitle('Arizona State Aspect Sentiments PDP\n'
             'Partial Dependency of users review sentiments against their star ratings')
plt.subplots_adjust(top=0.9)
fig2 = plt.figure()
# plt.show()
fig2.show()

# NZ
predictoR3 = pdpNV[pdpNV.columns - ['Stars']]
targeT3 = pdpNV['Stars']
dataRS3 = random.sample(pdpNV.index, int(len(pdpNV)*.80))
predictorTrain3, targetTrain3 = predictoR3.ix[dataRS3],targeT3.ix[dataRS3]
predictorTest3, targetTest3  = predictoR3.drop(dataRS3),targeT3.drop(dataRS3)

dataGBR3 = GradientBoostingRegressor(n_estimators=1000, max_depth=4,
                                learning_rate=0.1, loss='huber',
                                random_state=1).fit(predictorTrain3, targetTrain3)


msE3 = mean_squared_error(targetTest3, dataGBR3.predict(predictorTest3))
R23 = r2_score(targetTest3, dataGBR3.predict(predictorTest3))

print("All MSE: %.4f" % msE3)
print("All R2: %.4f" % R23)


pltNameData = ['Food_Avg', 'Service_Avg', 'Price_Avg', 'Ambience_Avg']
pltData = [0, 1, 2, 3] # Which columns to take input from the predictors train dataset
fig3, axs = plot_partial_dependence(dataGBR3, predictorTrain3, pltData, feature_names=pltNameData, n_jobs=1, n_cols=2, grid_resolution=50)

fig3.suptitle('Nevada State Aspect Sentiments PDP\n'
             'Partial Dependency of users review sentiments against their star ratings')
plt.subplots_adjust(top=0.9)
fig3 = plt.figure()
# plt.show()
fig3.show()