#pandas as a library
import pandas as pd
#sklearn is a library used in machine learning
from sklearn.model_selection import train_test_split
#Read a csv file,csv file assigned to dataset variable
dataset = pd.read_csv("Salary_Data.csv")
print(dataset)
#independent is a input variable;we get the yoe column alone
independent = dataset[["YearsExperience"]]
print(independent)
#dependent is output variable;we get salary part alone
dependent = dataset[["Salary"]]
print(dependent)
#Now its assigning csv data values to x_train,X_test,y-train,y_test;X is the input variable,y is the output variable
X_train,X_test,y_train,y_test = train_test_split(independent,dependent,test_size=0.30,random_state=0)
print(X_train,X_test,y_train,y_test)
#Model creation process
#its Linear regression part,regressor is a used tp predict the response variable

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
print(regressor.fit(X_train,y_train))
#to get weight constant value
weight = regressor.coef_
print(weight)
#to get bias constant value
bais =  regressor.intercept_
print(bais)
y_pred=regressor.predict(X_test)
#to find evaluation metrics-cross checking process using R square
from sklearn.metrics import r2_score
r_score = r2_score(y_test,y_pred)
print(r_score)
#save the model we using pickle
import pickle
#filename is stored in a variable picle name extension.sav
filename = "finalized_model_linear.sav"
#we are writting in the file or assingning the file

pickle.dump(regressor,open(filename,'wb'))
#Read the file
loaded_model = pickle.load(open("finalized_model_linear.sav",'rb'))
result =  loaded_model.predict([[13]])
print(result)