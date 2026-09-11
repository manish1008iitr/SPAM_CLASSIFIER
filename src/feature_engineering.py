import pandas as pd 
import numpy as np
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

def load_params(params_path:str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_data(data_path:str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.dropna()
    return df

def apply_tfidf(train_data:pd.DataFrame, test_data:pd.DataFrame, max_features:int) -> tuple:
    vectorizer = TfidfVectorizer(
        max_features = max_features
    )
    X_train = train_data["text"]
    y_train = train_data["target"]
    X_test = test_data["text"]
    y_test = test_data["target"]

    X_train_vectorised = vectorizer.fit_transform(X_train)
    X_test_vectorised = vectorizer.transform(X_test)

    train_df = pd.DataFrame(X_train_vectorised.toarray())
    train_df["label"] = y_train 

    test_df = pd.DataFrame(X_test_vectorised.toarray())
    test_df["label"] = y_test

    return train_df, test_df

def save_data(df: pd.DataFrame, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

def main():
    params = load_params(params_path='../params.yaml')
    max_features = params['feature_engineering']['max_features']
    

    train_data = load_data('../data/processed/processes_train_data.csv')
    test_data = load_data('../data/processed/processes_test_data.csv')

    # Vectorize the data
    vectorised_data = apply_tfidf(train_data, test_data, max_features)

    train_vectorised_data =    vectorised_data[0]
    test_vectorised_data = vectorised_data[1]
    #test_vectorised_data = apply_tfidf(test_data, test_data, max_features)

    save_data(train_vectorised_data, os.path.join("../data", "vectorized", "train_tfidf.csv"))
    save_data(test_vectorised_data, os.path.join("../data", "vectorized", "test_tfidf.csv"))

if __name__ == '__main__':
    main()



