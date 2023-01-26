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
#import seaborn as sn
from sklearn import preprocessing
import copy
import requests
import json
import seaborn as sns
from numerize.numerize import numerize

st.title("Comparaisons")
st.sidebar.image("pret_a_depenser.png")


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

f = open("g","rb")
g = load(f)
f.close()

f = open("w","rb")
numero_client = load(f)
f.close()

f = open("model","rb")
model = load(f)
f.close()

xtrain = x 
ytest_2 = y
xtest_4 = z
modele = r
xtest_final = t
max_seuil = u
train_set_proba = v
xtest_2 = s
features = g

data = xtest_final
data_2 = xtest_final

data_un_client = xtest_4[xtest_4['SK_ID_CURR'] == numero_client]

def get_pret():
    data_un_client = xtest_4[xtest_4['SK_ID_CURR'] == numero_client]
    # Use the loaded pickled model to make predictions
    y_pred = model.predict(data_un_client[features])
   
    accord = 'Le prêt est accordé ' 
    refuse = 'Le prêt est refusé  ' 
    
    if y_pred[0] == 0 : 
        return accord 
    else : return  refuse
    
    
def get_pret_data():
    
    # Use the loaded pickled model to make predictions
    y_pred_data = model.predict(xtest_4[features])
    
    dict_pred = dict()
    for i in range(xtest_4.shape[0]):
        
        num_client =int(xtest_4.at[i,'SK_ID_CURR' ])
        dict_pred [num_client]=  int(y_pred_data[i])
        print(y_pred_data[i])  
    return  dict_pred


result = get_pret()#requests.get("http://127.0.0.1:5000/get_pret/")
result_2 = get_pret_data()#requests.get("http://127.0.0.1:5000/get_pret_data/")
dict_new = result_2#.json()



data_2['predict'] = pd.Series(dict_new.values())



option = st.selectbox(
    'Variables à comparer',
    ('AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    ))

st.write('You selected:', option)

valeurs_option = data[option].value_counts()

fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    df_2 = data_2[data_2['predict']== 0]
    df_3 = data_2[data_2['predict']== 1]
    df_4 = data[data['SK_ID_CURR']== numero_client]
    fig_col1, ax = plt.subplots()
    
    ax = sns.kdeplot(data=df_2, x='AMT_INCOME_TOTAL', hue = 'TARGET')
    ax  = sns.kdeplot(data=df_3, x='AMT_INCOME_TOTAL', hue = 'TARGET')    
    ax = plt.axvline(x=float(df_4['AMT_INCOME_TOTAL']), ymin = 0, ymax = 1, linewidth=2, color='r')

    st.pyplot(fig_col1)

with fig_col2:
    df_2 = data_2[data_2['predict']== 0]
    df_3 = data_2[data_2['predict']== 1]
    df_4 = data[data['SK_ID_CURR']== numero_client]
    fig_col2, ax2 = plt.subplots()
     
    ax2 = sns.kdeplot(data=df_2, x='AMT_CREDIT', hue = 'TARGET') 
    ax2  = sns.kdeplot(data=df_3, x='AMT_CREDIT', hue = 'TARGET')
    
    ax2 = plt.axvline(x=float(df_4['AMT_CREDIT']), ymin = 0, ymax = 1, linewidth=2, color='r')
    
    st.pyplot(fig_col2)
    
option_2 = st.sidebar.selectbox(
    'Variables à comparer',
    (list(data_2['NAME_EDUCATION_TYPE'].unique())
     ))

option_3 = st.sidebar.selectbox(
    'Variables à comparer',
    (list(data_2['NAME_FAMILY_STATUS'].unique())
     ))

option_4 = st.sidebar.selectbox(
    'Variables à comparer',
    (list(data_2['NAME_HOUSING_TYPE'].unique())
     ))


df_education = data_2[data_2['NAME_EDUCATION_TYPE'] == option_2 ]
df_family = data_2[data_2['NAME_FAMILY_STATUS'] == option_3 ]
df_house = data_2[data_2['NAME_HOUSING_TYPE'] == option_4 ]

fig_col3, fig_col4, fig_col5 = st.columns(3)

with fig_col3:
    df_education_final = df_education['predict'].value_counts()
    labels = df_education['predict'].unique()
    fig_col3, ax4 = plt.subplots()
    ax4.pie(df_education_final,  autopct='%.2f%%')
    ax4.legend(labels)   
    ax4.set_title("Predict Name education")   
    st.pyplot(fig_col3)

with fig_col4:
    df_family_final = df_family['predict'].value_counts()
    fig_col4, ax = plt.subplots()
    ax.pie(df_family_final,  autopct='%.2f%%') 
    ax.legend(labels)   
    ax.set_title("Predict Family status")   
    st.pyplot(fig_col4)


with fig_col5:
    df_house_final = df_house['predict'].value_counts()
    fig_col5, ax = plt.subplots()
    ax.pie(df_house_final,  autopct='%.2f%%') 
    ax.legend(labels)   
    ax.set_title("Predict House Type")   
    st.pyplot(fig_col5)
