# data preprocessing

import pandas as pd
import os
import nltk
import logging
from utility.utils import *

# logging configuration
logger = logging.getLogger("data_transformation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("transformation_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download("wordnet")
nltk.download("stopwords")


def normalize_text(df):
    """Normalize the text data."""
    try:
        df.loc[:, "content"] = df.loc[:, "content"].apply(remove_small_sentences)
        logger.debug("removing small sentences which has less than 3 words")
        df.loc[:, "content"] = df.loc[:, "content"].apply(lower_case)
        logger.debug("converted to lower case")
        df.loc[:, "content"] = df.loc[:, "content"].apply(remove_stop_words)
        logger.debug("stop words removed")
        df.loc[:, "content"] = df.loc[:, "content"].apply(removing_numbers)
        logger.debug("numbers removed")
        df.loc[:, "content"] = df.loc[:, "content"].apply(removing_punctuations)
        logger.debug("punctuations removed")
        df.loc[:, "content"] = df.loc[:, "content"].apply(removing_urls)
        logger.debug("urls")
        df.loc[:, "content"] = df.loc[:, "content"].apply(lemmatization)
        logger.debug("lemmatization performed")
        logger.debug("Text normalization completed")
        return df
    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("data loaded properly")

        # Transform the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug("Processed data saved to %s", data_path)
    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
