import re
import os
import yaml
import string
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    """Remove numbers from the text."""
    text = "".join([char for char in text if not char.isdigit()])
    return text


def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = text.replace("؛", "")
    text = re.sub("\s+", " ", text).strip()
    return text


def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r"", text)


def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params
    except FileNotFoundError:
        raise
    except yaml.YAMLError as e:
        raise
    except Exception as e:
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        return df
    except pd.errors.ParserError as e:
        raise
    except Exception as e:
        raise "Unexpected error occurred while saving the data"


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        raise
    except Exception as e:
        raise
