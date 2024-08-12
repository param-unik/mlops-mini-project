# feature engineering

import os
import logging
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from utility.utils import load_data, save_data, load_params

# logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("feature_engineering_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply Count Vectorizer to the data."""
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data["content"].values
        y_train = train_data["sentiment"].values
        X_test = test_data["content"].values
        y_test = test_data["sentiment"].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df["label"] = y_test

        pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

        logger.debug("Bag of Words applied and data transformed")
        return train_df, test_df
    except Exception as e:
        logger.error("Error during Bag of Words transformation: %s", e)
        raise


def main():
    try:
        params = load_params("params.yaml")
        max_features = params["feature_engineering"]["max_features"]

        train_data = load_data("./data/interim/train_processed.csv")
        test_data = load_data("./data/interim/test_processed.csv")

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_bow.csv"))
    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
