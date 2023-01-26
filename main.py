import pickle
from pickle import *
import streamlit as st
from numerize.numerize import numerize
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import requests
import json
import plotly.express as px
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
from sklearn.ensemble import AdaBoostClassifier
#import seaborn as sn
from sklearn import preprocessing
import copy


st.set_page_config(
    page_title="Dashboard",
    page_icon="✅",
    layout="wide",
)

# dashboard title
#st.write('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"> ',  unsafe_allow_html=True)

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

f = open("g","rb")
g = load(f)
f.close()

f = open("s","rb")
s = load(f)
f.close()

f = open("model","rb")
model = load(f)
f.close()

#f = open("p","rb")
#p = load(f)
#f.close()


st.title("Prêt à dépenser")
st.sidebar.success('Selectionner une page')
st.sidebar.image("pret_a_depenser.png")

xtrain = x # avec features
ytest_2 = y
xtest_4 = z
modele = r
xtest_final = t
max_seuil = u
train_set_proba = v
xtest_2 = s
features = g

#xtest = p 

data = xtest_final

st.markdown("<h1 style='color: red;'>numero client : </h1>", unsafe_allow_html=True)
numero_client  = st.selectbox(
                        'Choissez le numéro du client',
                        data['SK_ID_CURR'].values)


f = open("w","wb")
w = numero_client
dump(w,f)
f.close() 


            

# ----- BEGIN : MAIN CODE -----
#On récupère l'index qui correspond au numéro client
ind = data[data['SK_ID_CURR']== numero_client].index.values.astype(int)[0]


#@app.route("/get_pret_data/", methods= ['GET'])
y_pred_data = model.predict(xtest_4[features])
    
dict_new = dict()
#for i in range(xtest_4.shape[0]):
        
#    num_client =int(xtest_4.at[i,'SK_ID_CURR' ])
#    dict_new [num_client]=  int(y_pred_data[i])
result = requests.get("https://api-scoring.herokuapp.com/get_pret_data/")#("http://127.0.0.1:5000/get_pret_data/")#
dict_new = result.json()
#dict_new = json.loads(result.text)
    
for j in range(len(dict_new) ):
    if int(data.at[ind,'SK_ID_CURR']) == int(list(dict_new.keys())[j]):
        data.at[ind,'predict'] = list(dict_new.values())[j]
               


#On affiche si le prêt est accordé ou pas

if data.at[ind,'predict'] == 0  : 
    st.sidebar.image("Pouce en l'air.png")
else : st.sidebar.image("Pouce en bas.png")


if data.at[ind,'predict'] == 0 : 
    st.subheader("Le crédit est accordé")
else : st.subheader("Le crédit est refusé")



#On affiche les information du client
fig_col5, fig_col6 = st.columns(2)

with fig_col5:
    st.write('Informations du client')
    fig5 = data.loc[ind]
    st.write(fig5)

#----------------Probabilité et seuil----------------------
data_2 = xtest_final
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

    
#---------------------------Seuil et probabilité---------------------
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




#--- On compare les données du client à la moyenne des données du fichier client. 
     
fig_col7, fig_col8, fig_col9, fig_col10, fig_col11 = st.columns(5) 

with fig_col7:#graphiste image Freepik
    nbre_d_enfants = data['CNT_CHILDREN'].mean()
    st.image('enfants.png',use_column_width='Auto')
    st.metric(label = 'Le nombre d\'enfants moyen est :', value=  numerize(nbre_d_enfants))


with fig_col8:
    salaire = data['AMT_INCOME_TOTAL'].mean()
    st.image('un-salaire.png',use_column_width='Auto')
    st.metric(label = 'Le salaire moyen est :', value=  numerize(salaire))

with fig_col9:
    emprunt = data['AMT_CREDIT'].mean()
    st.image('emprunter.png',use_column_width='Auto')
    st.metric(label = 'L \'emprunt moyen est :', value=  numerize(emprunt))

with fig_col10:
    age_moyen = data['DAYS_BIRTH'].mean()
    st.image('age.png',use_column_width='Auto')
    st.metric(label = 'L \'âge moyen relatif à la demande :', value=  numerize(age_moyen))

with fig_col11:
    jours_emploi = data['DAYS_EMPLOYED'].mean()
    st.image('perte-demploi.png',use_column_width='Auto')
    st.metric(label = 'Le nombre de jours d\'emploi  :', value=  numerize(jours_emploi))



fig_col12, fig_col13, fig_col14, fig_col15, fig_col16 = st.columns(5) 

with fig_col12:#graphiste image Freepik
    nbre_d_enfants = data.at[ind, 'CNT_CHILDREN']
    st.image('enfants-1.png',use_column_width='Auto')
    st.metric(label = 'Votre nombre d\'enfants est :', value=  nbre_d_enfants)


with fig_col13:
    salaire = data.at[ind,'AMT_INCOME_TOTAL']
    st.image('un-salaire-1.png',use_column_width='Auto')
    st.metric(label = 'Votre salaire est :', value=  numerize(salaire))

with fig_col14:
    emprunt = data.at[ind,'AMT_CREDIT']
    st.image('emprunter-1.png',use_column_width='Auto')
    st.metric(label = 'Votre emprunt est :', value=  numerize(emprunt))

with fig_col15:
    age = data.at[ind,'DAYS_BIRTH']
    st.image('age-1.png',use_column_width='Auto')
    st.metric(label = 'Votre \'âge  relatif à la demande :', value=  age)

with fig_col16:
    jours_emploi = data.at[ind,'DAYS_EMPLOYED']
    st.image('perte-demploi-1.png',use_column_width='Auto')
    st.metric(label = 'Le nombre de jours d\'emploi  :', value=  numerize(jours_emploi))


fig_col17 ,fig_col20  = st.columns(2) 
#

with fig_col17:
#graphiste image Freepik
    figure, ax = plt.subplots()
    ax.hist(data['NAME_EDUCATION_TYPE'])    
    ax.set_title('NAME EDUCATION TYPE')
    ax.legend(title ='NAME EDUCATION TYPE' )
    ax.tick_params(labelrotation=45)  
                              
    st.pyplot(figure)

with fig_col20:
    education = data.at[ind,'NAME_EDUCATION_TYPE']   
st.write(education)


fig_col18, fig_col21 = st.columns(2) 

with fig_col18:
    fig18, ax1 = plt.subplots()
    #fig, ax = plt.subplots(figsize=(10, 10))
    #fig = px.hist(data['NAME_FAMILY_STATUS'], x='Name family status', y='Nombre')
        #Show bar plot 
    #st.plotly_chart(fig)
    ax1.hist(data['NAME_FAMILY_STATUS'])
    ax1.set_title('NAME FAMILY STATUS')
    ax1.tick_params(labelrotation=45)
      
    st.pyplot(fig18)


with fig_col21:
    family = data.at[ind,'NAME_FAMILY_STATUS']   
st.write(family)

fig_col19, fig_col22 = st.columns(2)

with fig_col19:
    fig19, ax = plt.subplots()
    ax.hist(data['NAME_HOUSING_TYPE'])
    ax.set_title('NAME HOUSING TYPE')
    ax.tick_params(labelrotation=45)
   
    st.pyplot(fig19)

with fig_col22:
    house = data.at[ind,'NAME_HOUSING_TYPE']   
st.write(house)

# ----- END : MAIN CODE -----