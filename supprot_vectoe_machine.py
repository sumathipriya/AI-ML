import pandas as pd
dataset = pd.read_csv("50_Startups.csv")
print(dataset)
dataset = pd.get_dummies(dataset,drop_first=True)
print(dataset)
print(dataset.columns)
independent = dataset[['R&D Spend', 'Administration', 'Marketing Spend',
       'State_Florida', 'State_New York']]
print(independent)

dependent = dataset[['Profit']]
print(dependent)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(independent,dependent,test_size=0.30,random_state=0)


#Standarization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test =  sc.transform(X_test)

#procedure,kernel
#kernel= linear ==>output should be in a linear format,
# solution may get linear or non linear also,going to result in linear
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
#model going t create
regressor.fit(X_train,y_train)

print(regressor.intercept_)
print(regressor.n_support_)
print(regressor.support_)


#evaluation
y_pred =  regressor.predict(X_test)
from sklearn.metrics import r2_score
r_score = r2_score(y_test,y_pred)
print(r_score)






