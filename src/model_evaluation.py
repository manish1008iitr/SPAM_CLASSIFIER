import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import yaml
from dvclive import Live


def load_params(params_path:str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model 

def load_data(file_path: str):
    df = pd.read_csv(file_path)
    return df

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

    metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

    return metrics_dict


def save_metrics(metrics: dict, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)

def main():
    params = load_params(params_path='params.yaml')
    clf = load_model('./models/model.pkl')
    test_data = load_data('./data/processed/test_tfidf.csv')
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    metrics = evaluate_model(clf, X_test, y_test)

if __name__ == '__main__':
    main()
