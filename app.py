import pandas as pd
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier


import operator


df = pd.read_pickle("./p7.pkl")
df.head()


df.DAYS_BIRTH = df.DAYS_BIRTH.apply(lambda x: abs(int(x/365)))
df = df.rename(columns={"DAYS_BIRTH": "AGE"})
num_columns = df.select_dtypes(include=["float64"]).columns
transf = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL']
for var in transf:
    df[var] = np.exp(df[var]).astype(int)


X = df.iloc[:, 0:9]
y = df.iloc[:, 10]
rf = RandomForestClassifier(max_depth=25, min_samples_leaf=1, n_estimators=150)
rf = rf.fit(X, y)


def probability(X):
    probas = rf.predict_proba(X)
    probas = [proba[0] for proba in probas]
    return probas


df["RF_PROBA"] = probability(X)
df["RF_PRED"] = rf.predict(X)
df['Solvable'] = df["RF_PROBA"]
df['Non Solvable'] = 1-df["RF_PROBA"]


# Interprétabilité du modèle
lime_explainer = LimeTabularExplainer(X,
                                      feature_names=X.columns,
                                      discretize_continuous=False)


nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(X)


external_stylesheets = [dbc.themes.LUX]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([

    dcc.Tabs([
        # Premier onglet
        dcc.Tab(label='Solvabilité Client', children=[

            html.Div([
                html.H3("Id Client"),
                dcc.Dropdown(
                    id='id-client',
                    options=[{'label': i, 'value': i} for i in X.index],
                    value=X.index[0]
                ),
            ], className="m-5"),
            html.Div([

                html.Div([
                    html.H3("Probabilité de Solvabilité"),
                    dcc.Graph(id='proba',
                              figure={},
                              style={"height": 500,
                                     "width": 500}
                              ),
                ]),

                html.Div([
                    html.H3("Importance des paramètres pour le client"),
                    dcc.Graph(id='graph',
                              figure={},
                              style={"height": 500,
                                     "width": 800}
                              ),
                ], className=''),
            ], className="m-5"),

            html.Div([
                html.H3("Profil du client")], className="mx-5"),

            html.Div(id='table_client',

                     className='m-5'),


            html.Div([
                html.H3("Profils de clients similaires")], className="mx-5"),

            html.Div(id='table_clients',

                     className='m-5'),

        ]),

        # Deuxième onglet
        dcc.Tab(label='Exploration des données', children=[
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='xaxis-column',
                        options=[{'label': i, 'value': i}
                                 for i in num_columns],
                        value='AMT_CREDIT'
                    ),
                    dcc.RadioItems(
                        id='xaxis-type',
                        options=[{'label': i, 'value': i}
                                 for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ],
                    style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Dropdown(
                        id='yaxis-column',
                        options=[{'label': i, 'value': i}
                                 for i in num_columns],
                        value='AMT_ANNUITY'
                    ),
                    dcc.RadioItems(
                        id='yaxis-type',
                        options=[{'label': i, 'value': i}
                                 for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),

            dcc.Graph(id='indicator-graphic'),

        ]),
    ]),
])


@app.callback(
    Output('table_client', 'children'),
    [
        Input('id-client', 'value')])
def update_table(id_client):
    dff = X[X.index == id_client]

    return [dash_table.DataTable(
        id="table",
        columns=[
            {"name": i, "id": i} for i in X.columns
        ],
        data=dff.to_dict('records'))]


@app.callback(
    Output('table_clients', 'children'),
    [
        Input('id-client', "value")])
def update_table2(id_client):

    indices_similary_clients = nbrs.kneighbors(
        np.array(X.loc[id_client]).reshape(1, -1))[1].flatten()

    dff = X[X.index.isin(X.index[indices_similary_clients])]

    return [dash_table.DataTable(
        id="table2",
        columns=[
            {"name": i, "id": i} for i in X.columns
        ],
        data=dff.to_dict('records'))]


@app.callback(
    Output('proba', 'figure'),
    [Input('id-client', 'value')])
def proba_pie(id_client):
    values = df.loc[id_client]
    values = (values['Solvable'], values['Non Solvable'])

    return {
        'data': [go.Pie(labels=['Solvable', "Non Solvable"],
                        values=values,
                        marker_colors=["#2ecc71", "#e74c3c"],
                        hole=.5
                        )],
        'layout': go.Layout(margin=dict(b=100)
                            )
    }
    del values


@app.callback(
    Output('graph', 'figure'),
    [Input('id-client', 'value'),
     ])
def update_graphic(id_client):
    exp = lime_explainer.explain_instance(X.loc[id_client].values,
                                          rf.predict_proba,
                                          num_features=10)

    indices, values = [], []

    for ind, val in sorted(exp.as_list(), key=operator.itemgetter(1)):
        indices.append(ind)
        values.append(val)
    dat = pd.DataFrame(values, columns=["values"], index=indices)
    dat["positive"] = dat["values"] > 0
    del indices, values

    return {

        'data': [go.Bar(
            x=dat["values"],
            y=dat.index,
            orientation='h',
            marker_color=list(dat.positive.map(
                {True: '#e74c3c', False: '#2ecc71'}).values)
        )],

        'layout': go.Layout(
            margin=dict(l=300, r=0, t=30, b=100)
        )
    }


@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('xaxis-type', 'value'),
     Input('yaxis-type', 'value')])
def update_graph_2(xaxis_column_name, yaxis_column_name,
                   xaxis_type, yaxis_type):

    traces = []
    solvable_labels = ["Solvable", "Non Solvable"]
    for i, target in enumerate(df.RF_PRED.unique()):
        filtered_df = df[df['RF_PRED'] == target].reset_index()
        traces.append(dict(
            x=filtered_df[xaxis_column_name],
            y=filtered_df[yaxis_column_name],
            text=filtered_df.index,
            mode='markers',
            opacity=0.7,
            marker={
                'color': list(filtered_df["RF_PRED"].map({0.0: '#e74c3c', 1.0: "#2ecc71"}).values),
                'size': 5,
                'line': {'width': 0.15, 'color': 'white'}
            },
            name=solvable_labels[i]
        ))

    return {
        'data': traces,
        'layout': dict(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=False)
