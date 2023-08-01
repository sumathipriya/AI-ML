import  pickle
#Deployement phase
loaded_model = pickle.load(open("finalized_model_linear.sav",'rb'))
#0 is origin or initial value or bias value,its the intial salary 
result =  loaded_model.predict([[0]])
print(result)