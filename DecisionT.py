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

from sklearn.tree import DecisionTreeRegressor
regressor =DecisionTreeRegressor()
regressor = regressor.fit(X_train,y_train)

import matplotlib.pyplot as plt
from sklearn import tree
tree.plot_tree(regressor)
plt.show()

y_pred =  regressor.predict(X_test)
from sklearn.metrics import r2_score
r_score = r2_score(y_test,y_pred)
print(r_score)