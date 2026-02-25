import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import FEATURES, MODEL_PARAMS, TEST_SIZE, RANDOM_STATE, FIELD_SIZE, DEFAULT_MODEL_PATH


def train(df, model_path=None, show_plots=True):
    """
    Train a RandomForestClassifier on the provided dataset.

    Prints accuracy and classification report. Optionally displays
    confusion matrix and feature importance plots. Saves the trained
    model to disk.

    Returns the trained classifier.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    X = df[FEATURES]
    y = df['made_tournament']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    clf = RandomForestClassifier(**MODEL_PARAMS)
    clf.fit(X_train, y_train)

    # evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    if show_plots:
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=sorted(set(df['made_tournament']))
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

        # feature importance
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = X.columns

        plt.figure(figsize=(8, 6))
        plt.title("Feature Importances")
        plt.barh(range(X.shape[1]), importances[indices], align='center')
        plt.yticks(range(X.shape[1]), feature_names[indices])
        plt.ylabel("Feature")
        plt.xlabel("Importance")
        plt.show()

    # save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f'Model saved to {model_path}')

    return clf


def load_model(model_path=None):
    """Load a trained model from disk."""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    return joblib.load(model_path)


def predict(clf, df):
    """
    Predict tournament field by ranking top 68 teams per year
    by predict_proba.

    Returns a copy of df with tournament_prob and predicted_tournament columns.
    """
    X = df[FEATURES]
    probabilities = clf.predict_proba(X)[:, 1]

    df = df.copy()
    df['tournament_prob'] = probabilities
    df['predicted_tournament'] = (
        df.groupby('year')['tournament_prob']
        .rank(method='first', ascending=False) <= FIELD_SIZE
    )

    return df
