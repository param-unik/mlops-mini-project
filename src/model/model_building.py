# model building

import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from utility.utils import load_data, save_model

# logging configuration
logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_building_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=1, solver="liblinear", penalty="l2")
        clf.fit(X_train, y_train)
        logger.debug("Model training completed")
        return clf
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise


def main():
    try:
        train_data = load_data("./data/processed/train_bow.csv")
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)

        save_model(clf, "models/model.pkl")
    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
