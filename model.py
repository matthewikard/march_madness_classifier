import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import FEATURES, MODEL_PARAMS, MODEL_TYPE, TEST_SIZE, RANDOM_STATE, FIELD_SIZE, DEFAULT_MODEL_PATH, model_path_for


CLASSIFIERS = {
    'random_forest': RandomForestClassifier,
    'logistic_regression': LogisticRegression,
    'gradient_boosting': GradientBoostingClassifier,
}


def _build_model(model_type):
    """Build a classifier, wrapping in a scaling pipeline if needed."""
    if model_type not in CLASSIFIERS:
        raise ValueError(f'Unknown model type: {model_type}. Choose from: {list(CLASSIFIERS.keys())}')

    params = MODEL_PARAMS.get(model_type, {})
    clf = CLASSIFIERS[model_type](**params)

    # Logistic regression needs feature scaling
    if model_type == 'logistic_regression':
        return Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    return clf


def train(df, model_path=None, show_plots=True, model_type=None):
    """
    Train a classifier on the provided dataset.

    Prints accuracy and classification report. Optionally displays
    confusion matrix and feature importance plots. Saves the trained
    model to disk.

    Returns the trained classifier.
    """
    if model_type is None:
        model_type = MODEL_TYPE
    if model_path is None:
        model_path = model_path_for(model_type)

    X = df[FEATURES]
    y = df['made_tournament']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f'Model type: {model_type}')
    clf = _build_model(model_type)
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
        plt.title(f"Confusion Matrix ({model_type})")
        plt.show()

        # feature importance (not available for all model types via pipeline)
        _plot_feature_importance(clf, X.columns, model_type)

    # save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f'Model saved to {model_path}')

    return clf


def _plot_feature_importance(clf, feature_names, model_type):
    """Plot feature importances or coefficients depending on model type."""
    if model_type == 'logistic_regression':
        # Pipeline: get the inner classifier's coefficients
        inner = clf.named_steps['clf'] if hasattr(clf, 'named_steps') else clf
        importances = np.abs(inner.coef_[0])
        title = "Feature Coefficients (absolute value)"
    elif model_type in ('random_forest', 'gradient_boosting'):
        importances = clf.feature_importances_
        title = "Feature Importances"
    else:
        return

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.title(f"{title} ({model_type})")
    plt.barh(range(len(feature_names)), importances[indices], align='center')
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    plt.show()


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
