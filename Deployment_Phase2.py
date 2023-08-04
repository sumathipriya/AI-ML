import pickle
#Deployment Phase
loaded_model =pickle.load(open("finalized_model_Mul_linear.sav",'rb'))
result =loaded_model.predict([[1234,345,4565,1,0]])
print(result)