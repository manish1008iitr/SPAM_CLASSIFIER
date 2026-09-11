import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

from pathlib import Path
script_dir = Path(__file__).parent

def load_params(params_path:str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_data(data_path:str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.dropna()
    return df

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
    clf.fit(X_train, y_train)
    return clf

def save_model(model, dir_name, file_path: str) -> None:
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def main():
    params = load_params(script_dir.parent / "params.yaml")['model_building']
    train_data = load_data(script_dir.parent / "data/vectorized/train_tfidf.csv")
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    clf = train_model(X_train, y_train, params)    
    model_save_path = script_dir.parent / "models/model.pkl"
    save_model(clf, '../models', model_save_path)

if __name__ == '__main__':
    main()

