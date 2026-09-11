import pandas as pd
import os 
import yaml 
from sklearn.model_selection import train_test_split

def load_params(params_path:str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def data_loader(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df 

def data_processor(df:pd.DataFrame) -> pd.DataFrame:
    df.rename(columns = {
        'v1': 'target', 
        'v2': 'text'
        }, inplace = True)
    df = df[["target","text"]]
    return df

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path:str) -> None:
    raw_data_path = os.path.join(data_path,"raw")
    os.makedirs(raw_data_path, exist_ok=True)
    train_data.to_csv(os.path.join(raw_data_path,"train.csv"))
    test_data.to_csv(os.path.join(raw_data_path,"test.csv"))

def main():
    params = load_params(r"../params.yaml")
    test_size = params["data_ingestion"]["test_size"]
    data = data_loader("../experiments/spam.csv") 
    processed_data = data_processor(data)

    train_data, test_data = train_test_split(
        processed_data, test_size = test_size, random_state = 42
    )

    save_data(train_data, test_data, data_path=r"../data")

if __name__ == "__main__":
    main()
        