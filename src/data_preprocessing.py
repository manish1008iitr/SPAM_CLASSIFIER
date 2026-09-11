import os 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder 
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords 
import string 
import nltk 
nltk.download("stopwords")
nltk.download("punkt_tab")

def tranform_text(text):
    ps = PorterStemmer()
    text = text.lower()

    #tokenization of the text 
    text = nltk.word_tokenize(text)

    # Remove non-alphanumerc token 
    text = [word for word in text if word.isalnum()]

    # Removing punctuation and stopwords
    text = [word for word in text if word not in stopwords.words("english") and word not in string.punctuation]

    #Now stem the words 

    text = [ps.stem(word) for word in text]

    return " ".join(text)

def preprocess(df):
    # THe function is to preprocess the target column by removing duplicates, and tranforming the text column 

    encoder = LabelEncoder()
    df["target"] = encoder.fit_tranform(df["target"])
    return df

def main():
    train_data = pd.read_csv(r"C:\Users\Manish kumar\OneDrive\Documentos\SPAM_CLASSIFIER\src\data\_raw\train.csv") 
    test_data = pd.read_csv(r"C:\Users\Manish kumar\OneDrive\Documentos\SPAM_CLASSIFIER\src\data\_raw\test.csv")


    # Calling the above function to tranform the text column 
    train_data = train_data.loc[:,"text"].apply(tranform_text)
    test_data = test_data.loc[:,"text"].apply(tranform_text)

    train_data = train_data.iloc[:,-1].apply(preprocess)
    test_data = train_data.iloc[:,-1].apply(preprocess)

    data_path = os.path.join("./data","interim")
    os.makedirs(data_path, exist_ok = True)

    train_data.to_csv(os.path.join(data_path,"processes_train_data.csv"))
    test_data.to_csv(os.path.join(data_path,"processes_test_data.csv"))


if __name__ == "__main__":
    main()

