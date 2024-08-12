# hyperparameter tuning for Lor with bow vectorizer

# Import necessary libraries
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import dagshub

sys.path.append(str(Path(__file__).parent.parent))
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

    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["content"])
    y = df["sentiment"].astype("int8")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set the experiment name
    mlflow.set_experiment("LoR Hyperparameter Tuning")

    # Define hyperparameter grid for Logistic Regression
    param_grid = {"C": [0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"]}

    # Start the parent run for hyperparameter tuning
    with mlflow.start_run():

        lreg = LogisticRegression()

        # Perform grid search
        grid_search = GridSearchCV(lreg, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Log each parameter combination as a child run
        for params, mean_score, std_score in zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"],
        ):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)

                # Model evaluation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Log parameters and metrics
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_score", mean_score)
                mlflow.log_metric("std_cv_score", std_score)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Print the results for verification
                print(f"Mean CV Score: {mean_score}, Std CV Score: {std_score}")
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")

        # Log the best run details in the parent run
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_score)

        print(f"Best Params: {best_params}")
        print(f"Best F1 Score: {best_score}")

        # Save and log the notebook
        mlflow.log_artifact(__file__)

        # Log model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")


if __name__ == "__main__":

    # setting up the things like mlfow and dagshub
    setup()

    # load the dataset
    df = load_data()

    # run the main process
    main(df)
