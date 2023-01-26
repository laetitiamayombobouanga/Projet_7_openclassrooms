import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import plotly.graph_objects as go
import random
import lime
from lime import lime_tabular
import warnings
import pickle
from pickle import *
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn import preprocessing
import copy
import requests
import json
#import seaborn as sns
from numerize.numerize import numerize

st.title("Probabilité et seuil")
st.sidebar.image("pret_a_depenser.png")


f = open("d","rb")
data_2 = load(f)
f.close()

f = open("w","rb")
numero_client = load(f)
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
xtest_final= t
max_seuil = u
train_set_proba = v
xtest = p 
xtest_2 = s
features = g

data = xtest_final
data_un_client = xtest_4[xtest_4['SK_ID_CURR'] == numero_client]
st.set_option('deprecation.showPyplotGlobalUse', False)

def get_probabilite():
    
    #y_proba = model.predict_proba(data_un_client[features])
    y_proba_num = model.predict_proba(data_un_client[features])[:,0]#y_proba[0]
    return str(y_proba_num)

result_3 = get_probabilite()#requests.get("http://127.0.0.1:5000/get_probabilite/")


probability =  float(result_3[1:7])#.text[1:7])
seuil = max_seuil



st.write("Le seuil est :" , seuil)
st.write("La probabilité d'avoir la classe 0 ou le crédit accepté est de : :" , probability)

    

fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = probability,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Probabilité", 'font': {'size': 24}},
    delta = {'reference': probability, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkgreen"},
            'bar': {'color': "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 250], 'color': 'cyan'},
                {'range': [250, probability], 'color': 'royalblue'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': seuil}}))

st.plotly_chart(fig, use_container_width=True)



st.markdown('Ce graphique montre l`impact négatif (rouge) et positif (vert) sur la probabilité')

    # Les features qui expliquent la décision(LIME):
explainer = lime_tabular.LimeTabularExplainer(xtrain.values, mode="classification",
                                                 class_names= data['TARGET'].unique(),
                                                feature_names= xtrain.columns.tolist(),
                                             )

idx = random.randint(1, len(xtest))
class_names= data['TARGET'].unique()
feature_names= xtrain.columns.tolist()



explanation = explainer.explain_instance(xtest_2[idx], modele.predict_proba,
                                         num_features=20)#len(X_sm_train.columns.tolist()))


def main():
	
	explanation.as_pyplot_figure()    
	st.pyplot()

if __name__ == '__main__':
	main()
