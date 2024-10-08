# bow vs tfidf

# Import necessary libraries
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import dagshub

# sys.path.append(str(Path(__file__).parent.parent))
from utility.utils import (
    lemmatization,
    remove_stop_words,
    removing_numbers,
    removing_punctuations,
    removing_urls,
    lower_case,
)


def setup():

    mlflow.set_tracking_uri("https://dagshub.com/param-unik/mlops-mini-project.mlflow")
    dagshub.init(repo_owner="param-unik", repo_name="mlops-mini-project", mlflow=True)


def load_data():
    # Load the data
    df = pd.read_csv(
        "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
    ).drop(columns=["tweet_id"])

    print(df.head())

    return df


def normalize_text(df):
    """Normalize the text data."""
    try:
        df.loc[:, "content"] = df.loc[:, "content"].apply(lower_case)
        df.loc[:, "content"] = df.loc[:, "content"].apply(remove_stop_words)
        df.loc[:, "content"] = df.loc[:, "content"].apply(removing_numbers)
        df.loc[:, "content"] = df.loc[:, "content"].apply(removing_punctuations)
        df.loc[:, "content"] = df.loc[:, "content"].apply(removing_urls)
        df.loc[:, "content"] = df.loc[:, "content"].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise


def main(df):

    x = df.loc[:, "sentiment"].isin(["happiness", "sadness"])
    df = df[x]

    df.loc[:, "sentiment"] = df.loc[:, "sentiment"].map({"sadness": 0, "happiness": 1})

    # Normalize the text data
    df = normalize_text(df)
    print("After normalization...")
    print(df.head())

    # Set the experiment name
    mlflow.set_experiment("Bow vs TfIdf")

    # Define feature extraction methods
    vectorizers = {
        "BoW": CountVectorizer(),
        "TF-IDF": TfidfVectorizer(),
    }

    # Define algorithms
    algorithms = {
        "LogisticRegression": LogisticRegression(),
        "MultinomialNB": MultinomialNB(),
        "XGBoost": XGBClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
    }

    # Start the parent run
    with mlflow.start_run(run_name="All Experiments") as parent_run:

        # Loop through algorithms and feature extraction methods (Child Runs)
        for algo_name, algorithm in algorithms.items():
            for vec_name, vectorizer in vectorizers.items():
                with mlflow.start_run(
                    run_name=f"{algo_name} with {vec_name}", nested=True
                ) as child_run:
                    X = vectorizer.fit_transform(df["content"])
                    y = df["sentiment"].astype("int8")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    # Log preprocessing parameters
                    mlflow.log_param("vectorizer", vec_name)
                    mlflow.log_param("algorithm", algo_name)
                    mlflow.log_param("test_size", 0.2)

                    # Model training
                    model = algorithm
                    model.fit(X_train, y_train)

                    # Log model parameters
                    if algo_name == "LogisticRegression":
                        mlflow.log_param("C", model.C)
                    elif algo_name == "MultinomialNB":
                        mlflow.log_param("alpha", model.alpha)
                    elif algo_name == "XGBoost":
                        mlflow.log_param("n_estimators", model.n_estimators)
                        mlflow.log_param("learning_rate", model.learning_rate)
                    elif algo_name == "RandomForest":
                        mlflow.log_param("n_estimators", model.n_estimators)
                        mlflow.log_param("max_depth", model.max_depth)
                    elif algo_name == "GradientBoosting":
                        mlflow.log_param("n_estimators", model.n_estimators)
                        mlflow.log_param("learning_rate", model.learning_rate)
                        mlflow.log_param("max_depth", model.max_depth)

                    # Model evaluation
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    # Log evaluation metrics
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)

                    # Log model
                    mlflow.sklearn.log_model(model, "model")

                    # Save and log the notebook
                    mlflow.log_artifact(__file__)

                    # Print the results for verification
                    print(f"Algorithm: {algo_name}, Feature Engineering: {vec_name}")
                    print(f"Accuracy: {accuracy}")
                    print(f"Precision: {precision}")
                    print(f"Recall: {recall}")
                    print(f"F1 Score: {f1}")


if __name__ == "__main__":

    # setting up the things mlfow and dagshub
    setup()

    # load the dataset
    df = load_data()

    # run the main process
    main(df)
