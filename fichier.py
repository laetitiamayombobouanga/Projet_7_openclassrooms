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
#import seaborn as sns
from numerize.numerize import numerize

st.title("Fichier clients")
st.sidebar.image("pret_a_depenser.png")





f = open("t","rb")
t = load(f)
f.close()


#xtrain = x # avec features
#ytest_2 = y
#xtest_4 = z
#modele = r
xtest_final= t
#max_seuil = u
#train_set_proba = v
#xtest_2 = s

data_sample = xtest_final.sample(100)

if st.sidebar.checkbox("Afficher les données brutes", False):
    st.subheader("Jeu de données echantillon de 200 observations")
    st.write(data_sample)
