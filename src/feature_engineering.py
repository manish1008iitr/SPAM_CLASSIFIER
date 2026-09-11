import pandas as pd 
import os 
from sklearn.feature_extraction.text import TfidVectorizer
import yaml

def load_params(params_path:str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_data(data_path:str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df

def apply_tfidf(train_data:pd.DataFrame, test_data:pd.DataFrame, max_features:int) -> tuple:
    vectorizer = TfidVectorizer(
        max_features = max_features
    )

    X_train = train_data["test"].values
    y_train = train_data["target"].values
    X_test = test_data["test"].values
    y_test = test_data["target"].values

    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transfrom(X_test)

    train_df = pd.DataFrame(X_test_bow.toarray())
    train_df["lable"] = y_train 

    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df["label"] = y_test

    return train_df, test_df

def save_data(df: pd.DataFrame, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

def main():
    params = load_params(params_path='params.yaml')
    max_features = params['feature_engineering']['max_features']
    

    train_data = load_data('./data/interim/train_processed.csv')
    test_data = load_data('./data/interim/test_processed.csv')

    # Transform the data
    train_processed_data = apply_tfidf(train_data, "text", "test")
    test_processed_data = apply_tfidf(test_data, "test", "test")

    train_df, test_df = apply_tfidf(train_data, test_data, max_features)

    save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
    save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
        
    train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

if __name__ == '__main__':
    main()



