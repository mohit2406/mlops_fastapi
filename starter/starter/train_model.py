# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

from starter.ml.model import train_model, compute_model_metrics, inference, compute_slice_metrics
from starter.ml.data import process_data
from starter.ml.clean_data import basic_cleaning

# Add code to load in the data.

def main():
    # Load data
    data = pd.read_csv("data/census.csv")

    # Clean data
    cleaned_data, cat_cols, num_cols = basic_cleaning(data, "data/census_cleaned.csv", "salary")

    # split data into train and test
    train_full, test = train_test_split(cleaned_data, test_size=0.20)

    # Spli data into train and validation
    train, val = train_test_split(train_full, test_size=0.25)

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_cols, label="salary", training=True)
    X_val, y_val, _, _ = process_data(val, categorical_features=cat_cols, label="salary", training=False, encoder=encoder, lb=lb)

    # Train model.
    model = train_model(X_train, y_train)

    # Evaluate the model on the validation data.
    y_pred = inference(model, X_val)

    # Compute the model metrics on the validation data.
    precision, recall, fbeta = compute_model_metrics(y_val, y_pred)

    # Train model on the full training data.
    X_train_full, y_train_full, _, _ = process_data(train_full, categorical_features=cat_cols, label="salary", training=True)
    model = train_model(X_train_full, y_train_full)

    # Evaluate the model on the test data.
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_cols, label="salary", training=False, encoder=encoder, lb=lb)
    y_pred = inference(model, X_test)
    # Compute the model metrics on the test data.
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    # compute slice_feature (education) metrics
    slice_metrics = compute_slice_metrics(test, 'salary', cat_cols, 'education', model, encoder, lb)


if __name__ == "__main__":
    main()
