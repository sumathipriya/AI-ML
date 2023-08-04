import numpy as np
import pandas as pd
dataset = pd.read_csv("50_Startups.csv")
print(dataset)

#Its a nominal data convet into one hot encoding,column expansion is happen here,catogorical data convert in numerical
#ML algorithm should not allow dummies,so we drop, to avoid the repetation
dataset = pd.get_dummies(dataset,drop_first=True)
print(dataset)

print(dataset.columns)
independent = dataset[['R&D Spend', 'Administration', 'Marketing Spend',
       'State_Florida', 'State_New York']]
print(independent)

dependent = dataset[['Profit']]
print(dependent)
#input and output is splitted
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(independent,dependent, test_size=0.30,random_state=0)
#Model Creation
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
print(regressor.fit(X_train,y_train))
weight = regressor.coef_
print(weight)
bais = regressor.intercept_
print(bais)
#Evaluation
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
r_score = r2_score(y_test,y_pred)
print(r_score)
#Finalized model
import pickle
filename = "finalized_model_Mul_linear.sav"
pickle.dump(regressor,open(filename,'wb'))
#load the data
loaded_model = pickle.load(open("finalized_model_Mul_linear.sav",'rb'))
result = loaded_model.predict([[1234,345,4565,1,0]])
print(result)

