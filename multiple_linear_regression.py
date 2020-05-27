import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn 
seaborn.set()

raw_data = pd.read_csv('1.03. Dummies.csv')
data = raw_data.copy()

data['Attendance'] = data['Attendance'].map({'Yes':1, 'No':0})


data.describe()

y = data['GPA']
x1 = data[['SAT', 'Attendance']]


x = sm.add_constant(x1)
results =  sm.OLS(y,x).fit()
results.summary()

#plot data
plt.scatter(data['SAT'], y)
yhat_no = 0.643 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'], yhat_no, lw=2 , color = 'green')
fig = plt.plot(data['SAT'], yhat_yes, lw=2 , color = 'red')
plt.xlabel('SAT' , fontsize = '20')
plt.ylabel('GPA' , fontsize = '20')


#specific plot
plt.scatter(data['SAT'], y, c=data['Attendance'], cmap='RdYlGn_r')
yhat_no = 0.643 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'], yhat_no, lw=2 , color = 'green')
fig = plt.plot(data['SAT'], yhat_yes, lw=2 , color = 'red')
plt.xlabel('SAT' , fontsize = '20')
plt.ylabel('GPA' , fontsize = '20')

#scaterring the plot with dummies 0=yhat_no=no 1=yhat_yes=yes and yhat
plt.scatter(data['SAT'], y, c=data['Attendance'], cmap='RdYlGn_r')
yhat_no = 0.643 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
yhat = 0.0017*data['SAT'] + 0.275
fig = plt.plot(data['SAT'], yhat_no, lw=2 , c = 'green', label = 'regression line1')
fig = plt.plot(data['SAT'], yhat_yes, lw=2 , c = 'red', label = 'regression line2')
fig = plt.plot(data['SAT'], yhat, lw=3 , c = 'blue', label = 'regression line')
plt.xlabel('SAT' , fontsize = '20')
plt.ylabel('GPA' , fontsize = '20')
plt.show()



#predictions
x
new_data = pd.DataFrame({'const':1, 'SAT':[1700, 1670], 'Attendance':[0,1]})
new_data = new_data[['const', 'SAT', 'Attendance']]
new_data.rename(index={0:'Bob', 1:'Alice'})

predictions = results.predict(new_data)


predictionsdf = pd.DataFrame({'Predictions':predictions})
joined = new_data.join(predictionsdf)
joined.rename(index={0:'Bob', 1:'Alice'})
