# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:36:47 2021

@author: sarth
"""

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores - student_scores.csv')
dataset.head(10)

# Plotting the distribution of scores
dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()



X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values



from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) 



from sklearn.linear_model import LinearRegression  
reg = LinearRegression()  
reg.fit(X_train, y_train)



line = reg.coef_*X+reg.intercept_
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


print(X_test)
y_pred = reg.predict(X_test)    
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df



hours = [[9.25]]
own_pred = reg.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


from sklearn import metrics  
from sklearn.metrics import r2_score

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print('r2 score for model is : ', r2)