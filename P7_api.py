import operator

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors


st.title('Modele de scoring pour Pret a  dépenser')

@st.cache
def read_data(path):
    df = pd.read_pickle(path)
    return df

@st.cache
def format_df(df):
    df.DAYS_BIRTH = df.DAYS_BIRTH.apply(lambda x: abs(int(x/365)))
    df = df.rename(columns={"DAYS_BIRTH":"AGE"})
    transf = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL']
    for var in transf:
        df[var] = np.exp(df[var]).astype(int)
    df["SCORING_PROBABILITY"]=scoring_probability(slice_df(df)[0])
    df["MODEL_PREDICTION"]=model.predict(slice_df(df)[0])
    df['Solvable'] = df["SCORING_PROBABILITY"]
    df['Non Solvable']= 1-df["SCORING_PROBABILITY"]
    return df

@st.cache
def slice_df(df):
    X = df.iloc[:,0:9]
    y = df.iloc[:,10]
    return X, y
    
@st.cache(allow_output_mutation=True)
def fit_model(dataframe):
    model = RandomForestClassifier(max_depth=25, min_samples_leaf=1, n_estimators=150)
    model = model.fit(slice_df(dataframe)[0], slice_df(dataframe)[1])
    
    return model

@st.cache
def scoring_probability(X):
    probas = model.predict_proba(X)
    probas = [proba[0] for proba in probas]
    return probas

@st.cache
def clients_neighbors_data(X, id_client):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(X)
    indices_similary_clients = nbrs.kneighbors(
        np.array(X.loc[id_client]).reshape(1, -1))[1].flatten()
    df_clients = X[X.index.isin(X.index[indices_similary_clients])]
    return df_clients

@st.cache
def lime_chart(id_client, X):
    lime_explainer = LimeTabularExplainer(X,
                             feature_names=X.columns,
                             discretize_continuous=False)
    exp = lime_explainer.explain_instance(X.loc[id_client].values,
                                model.predict_proba,
                                num_features=10)
    indices, values = [], []
    
    for ind, val in sorted(exp.as_list(), key=operator.itemgetter(1)):
        indices.append(ind)
        values.append(val)
    dat = pd.DataFrame(values, columns=["values"], index=indices)
    dat["positive"] = dat["values"]>0
    fig = go.Figure([go.Bar(
                    x=dat["values"],
                    y=dat.index,
                    orientation='h',
                    marker_color=list(dat.positive.map({True: '#e74c3c', False: '#2ecc71'}).values))])
    return fig

@st.cache
def proba_pie(id_client):
    values = df_formatted.loc[id_client]
    values = (values['Solvable'],values['Non Solvable'])
    fig = go.Figure(data=[go.Pie(labels=['Solvable', "Non Solvable"],
                        values=values,
                        marker_colors=["#2ecc71", "#e74c3c"],
                        hole=.5
                       )])
    return fig

path = r"P7.pkl"

st.write("Uploading data ...")
df = read_data(path)
st.write("Data uploaded")
    
df = read_data(path)

st.write("Training is starting ...")
    
model = fit_model(df)
df_formatted = format_df(df)
    
st.subheader('Identifiant client')

client_id = st.selectbox("Choisissez un identifiant client", options=df.index)
st.write("Vous avez choisi le client", client_id)

st.subheader('Probabilite de solvabilite')
st.plotly_chart(proba_pie(client_id))

st.subheader('Informations client et clients similaires')
df_clients = clients_neighbors_data(slice_df(df)[0], client_id)
st.dataframe(df_clients)

st.subheader('Explicabilite avec Lime')
st.plotly_chart(lime_chart(client_id, slice_df(df)[0]))
