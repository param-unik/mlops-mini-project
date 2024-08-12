# data ingestion

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from utility.utils import load_params, load_data


# logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns=["tweet_id"], inplace=True)
        final_df = df[df["sentiment"].isin(["happiness", "sadness"])]
        final_df.loc[:, "sentiment"] = final_df.loc[:, "sentiment"].map(
            {"happiness": 1, "sadness": 0}
        )
        logger.debug("Data preprocessing completed")
        return final_df
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str
) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


def main():
    try:
        params = load_params(params_path="params.yaml")
        test_size = params["data_ingestion"]["test_size"]

        df = load_data(
            "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        )
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )
        save_data(train_data, test_data, data_path="./data")
    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
