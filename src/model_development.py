import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

def load_params(params_path:str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
    clf.fit(X_train, y_train)
    return clf

def save_model(model, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def main():
    params = load_params('params.yaml')['model_building']
    train_data = load_data('./data/processed/train_tfidf.csv')
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    clf = train_model(X_train, y_train, params)    
    model_save_path = 'models/model.pkl'
    save_model(clf, model_save_path)

