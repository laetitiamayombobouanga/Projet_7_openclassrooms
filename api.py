from flask import Flask, render_template, jsonify
import json
import requests
import pickle
from pickle import *

app = Flask(__name__)


f = open("w","rb")
w = load(f)
f.close()

f = open("model","rb")
model = load(f)
f.close()

f = open("x","rb")
x = load(f)
f.close()

f = open("y","rb")
y = load(f)
f.close()

f = open("z","rb")
z = load(f)
f.close()

f = open("r","rb")
r = load(f)
f.close()

f = open("t","rb")
t = load(f)
f.close()

f = open("u","rb")
u = load(f)
f.close()

f = open("v","rb")
v = load(f)
f.close()

f = open("s","rb")
s = load(f)
f.close()

f = open("p","rb")
p = load(f)
f.close()

f = open("g","rb")
g = load(f)
f.close()


xtrain = x # avec features
ytest_2 = y
xtest_4 = z
modele = r
xtest_final = t
max_seuil = u
train_set_proba = v
features = g
numero_client = w
xtest_2 = s

xtest_3 = p

data_un_client = xtest_4[xtest_4['SK_ID_CURR'] == numero_client]



@app.route("/get_pret/", methods= ['GET'])
def get_pret():
    data_un_client = xtest_4[xtest_4['SK_ID_CURR'] == numero_client]    
    # Use the loaded pickled model to make predictions
    y_pred = modele.predict(data_un_client[features])#
   
    accord = 'Le prêt est accordé ' 
    refuse = 'Le prêt est refusé  '
    #return str(data_un_client.shape[0])
    
    if y_pred[0] == 0 : 
        return accord 
    else : return  refuse
   


@app.route("/get_pret_data/", methods= ['GET'])#("https://app-scoring.herokuapp.com/get_pret_data", methods= ['GET'])#
def get_pret_data():
    
    # Use the loaded pickled model to make predictions
    y_pred_data = model.predict(xtest_4[features])
    
    dict_pred = dict()
    for i in range(xtest_4.shape[0]):
        
        num_client =int(xtest_4.at[i,'SK_ID_CURR' ])
        dict_pred [num_client]=  int(y_pred_data[i])
        print(y_pred_data[i])  
    return  dict_pred

@app.route("/get_probabilite/", methods= ['GET'])
def get_probabilite():
    
    y_proba = model.predict_proba(data_un_client[features])
    y_proba_num = y_proba[0]
    return str(y_proba_num)


if __name__ == "__main__":
    app.debug = True
    app.run()
    print("api start ! ")