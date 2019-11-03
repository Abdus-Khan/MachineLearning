import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

df = pd.read_csv('Fuel.csv', encoding = 'iso-8859-1')
df = df.head(50)

cdf = df[['EngineSize', 'Cylinders', 'FuelConsumption', 'CO2Emissions' ]]

msk = np.random.randn(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

from sklearn import linear_model

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['FuelConsumption']])
train_y = np.asanyarray(train[['CO2Emissions']])

regr.fit(train_x, train_y)

print('Coeficients:', regr.coef_)
print('Intercept: ',regr.intercept_)

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['FuelConsumption']])
test_y = np.asanyarray(test[['CO2Emissions']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

plt.scatter(train.FuelConsumption, train.CO2Emissions,  color='blue')
plt.plot(train_x, regr.coef_[0][0]* train_x + regr.intercept_[0], '-r')
plt.xlabel("FuelConsumption")
plt.ylabel("Emission")
plt.show()
