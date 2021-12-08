from typing import Optional
from fastapi import FastAPI
import pandas as pd
import numpy as np

from typing import List, Optional

from pydantic import BaseModel
from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier




df = pd.read_pickle("./p7.pkl")

df.DAYS_BIRTH = df.DAYS_BIRTH.apply(lambda x: abs(int(x/365)))
df = df.rename(columns={"DAYS_BIRTH": "AGE"})
num_columns = df.select_dtypes(include=["float64"]).columns
transf = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL']
for var in transf:
    df[var] = np.exp(df[var]).astype(int)


X = df.iloc[:, 0:9]
y = df.iloc[:, 10]
rf = RandomForestClassifier(max_depth=20, max_features=8, min_samples_leaf=10)
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

app = FastAPI()

class PaybackResponse(BaseModel):
    """
    the model which shows what is the response schema of the payback endpoint
    """
    client_id: int
    score: Optional[float] = None
    payback_probability: Optional[float] = None
    no_payback_probability: Optional[float] = None
    message: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "payback probability calculator"}


@app.get("/client/{client_id}", response_model=PaybackResponse)
def payback_probability(client_id: int):
    """
    Retourne les informations d'un client
    """
    try:
        values = df.loc[client_id]
        values = values.to_dict()
    except:
        return {"message": "Client not found", "client_id": client_id}
    return {
        "client_id": client_id,
        "score" :values["RF_PRED"],
        "payback_probability": values["Solvable"],
        "no_payback_probability": values["Non Solvable"]
    }
