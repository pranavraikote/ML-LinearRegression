# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:48:38 2018

@author: Pranav
"""

import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd


#Load data
df=pd.read_csv("Housing.csv")


Y = df['price']
X = df['lotsize']

X=X.reshape(len(X),1)
Y=Y.reshape(len(Y),1)

#Split into Training and Testing Data
X_train = X[:-250]
X_test = X[-250:]

Y_train = Y[:-250]
Y_test = Y[-250:]

#Plot a scatter graph
plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())
#plt.show()

#Create linear regression object
regr=linear_model.LinearRegression()
 
#Train the model using the training sets
regr.fit(X_train, Y_train)
 
#Plot outputs
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)
plt.show()
#Print coefficient and and intercept
print('Co-effiecient is {}'.format(regr.coef_))
print('Intercept is {}'.format(regr.intercept_)) 