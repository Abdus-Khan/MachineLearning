# import neccesary dependencies
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import pylab as pl 

#read the data set
df = pd.read_csv('Fuel.csv', encoding='iso-8859-1')
cdf = df[['EngineSize', 'Cylinders', 'FuelConsumption', 'CO2Emissions']]
cdf = cdf.head(50)

# Create mask so you can split dataset
msk = np.random.randn(len(cdf)) < 0.8
# Create training and testing data
train = cdf[msk]
test = cdf[~msk]

# start training your model by importing sklearn
from sklearn import linear_model
# call the linear regression model from sklearn
regr = linear_model.LinearRegression()
# Create x and y variables 
x = train[['EngineSize', 'Cylinders', 'FuelConsumption']]
y = train[['CO2Emissions']]
# fit our data to model
regr.fit(x, y)
# Create prediction 
y_hat = regr.predict(test[['EngineSize', 'Cylinders', 'FuelConsumption']])
x = test[['EngineSize', 'Cylinders', 'FuelConsumption']]
y = test[['CO2Emissions']]
# evaluate our model 
print('Residual sum of squares: %.2f' % np.mean((y_hat - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y)) 

