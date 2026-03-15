import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score

from config import FEATURES, SEED_FEATURES, MODEL_PARAMS, MODEL_TYPE, TEST_SIZE, RANDOM_STATE, FIELD_SIZE, DEFAULT_MODEL_PATH, model_path_for, seed_model_path_for


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


# ---------------------------------------------------------------------------
# Seed prediction model
# ---------------------------------------------------------------------------

def _build_seed_model():
    """Build a LinearRegression pipeline with StandardScaler for seed prediction."""
    return Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])


def train_seeds(df, model_path=None, show_plots=True):
    """
    Train a LinearRegression model to predict tournament seed (1-16).

    Only trains on teams where made_tournament is True and seed is not null.

    Returns the trained pipeline (StandardScaler + LinearRegression).
    """
    if model_path is None:
        model_path = seed_model_path_for()

    # Filter to tournament teams with known seeds
    tourney = df[df['made_tournament'] == True].copy()
    tourney = tourney[tourney['seed'].notna()]
    tourney['seed'] = tourney['seed'].astype(int)

    X = tourney[SEED_FEATURES]
    y = tourney['seed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f'Seed model: LinearRegression (scaling enabled)')
    print(f'Training on {len(X_train)} tournament teams, testing on {len(X_test)}')

    model = _build_seed_model()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MAE: {mae:.2f} seed lines')
    print(f'R-squared: {r2:.3f}')

    y_pred_rounded = np.clip(np.round(y_pred), 1, 16).astype(int)
    exact = (y_pred_rounded == y_test.values).mean()
    within_1 = (np.abs(y_pred_rounded - y_test.values) <= 1).mean()
    within_2 = (np.abs(y_pred_rounded - y_test.values) <= 2).mean()
    print(f'Exact seed match: {exact:.1%}')
    print(f'Within 1 seed line: {within_1:.1%}')
    print(f'Within 2 seed lines: {within_2:.1%}')

    if show_plots:
        # Actual vs predicted scatter
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.4, edgecolors='k', linewidths=0.5)
        plt.plot([1, 16], [1, 16], 'r--', label='Perfect')
        plt.xlabel('Actual Seed')
        plt.ylabel('Predicted Seed')
        plt.title('Seed Model: Actual vs Predicted')
        plt.legend()
        plt.show()

        # Feature coefficients
        inner_reg = model.named_steps['reg']
        coefs = np.abs(inner_reg.coef_)
        indices = np.argsort(coefs)[::-1]
        plt.figure(figsize=(8, 6))
        plt.title('Seed Model: Feature Coefficients (|coef|)')
        plt.barh(range(len(SEED_FEATURES)), coefs[indices], align='center')
        plt.yticks(range(len(SEED_FEATURES)), [SEED_FEATURES[i] for i in indices])
        plt.xlabel('|Coefficient|')
        plt.show()

    # Persist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f'Seed model saved to {model_path}')

    return model


def load_seed_model(model_path=None):
    """Load a trained seed prediction model from disk."""
    if model_path is None:
        model_path = seed_model_path_for()
    return joblib.load(model_path)


def predict_seeds(seed_model, field_df):
    """
    Assign tournament seeds (1-16) to the 68-team predicted field.

    Algorithm following NCAA bracket structure:
    1. Score all 68 teams with continuous predicted seed values.
    2. The 4 worst automatic qualifiers are First Four play-in teams (seed 16).
    3. The remaining 64 teams are assigned 4 per seed line (seeds 1-16).
    4. The 4 worst at-large teams are marked as First Four at-large play-in.

    Returns DataFrame with predicted_seed, predicted_seed_raw, and first_four columns.
    """
    df = field_df.copy()

    # Continuous seed prediction
    df['predicted_seed_raw'] = seed_model.predict(df[SEED_FEATURES])
    df['first_four'] = False
    df['predicted_seed'] = 0

    # --- First Four automatic qualifiers ---
    # The 4 worst AQs (highest predicted seed) play in the First Four as 16 seeds.
    aq_teams = df[df['is_aq'] == True].sort_values('predicted_seed_raw')
    if len(aq_teams) >= 4:
        ff_aq_idx = aq_teams.tail(4).index
        df.loc[ff_aq_idx, 'predicted_seed'] = 16
        df.loc[ff_aq_idx, 'first_four'] = True

    # --- Main bracket: remaining 64 teams ---
    remaining = df[df['predicted_seed'] == 0].sort_values('predicted_seed_raw')
    for i, idx in enumerate(remaining.index):
        df.loc[idx, 'predicted_seed'] = (i // 4) + 1

    # --- First Four at-large ---
    # The 4 worst at-large teams are First Four at-large play-in teams.
    al_teams = df[df['is_aq'] == False].sort_values('predicted_seed_raw')
    if len(al_teams) >= 4:
        ff_al_idx = al_teams.tail(4).index
        df.loc[ff_al_idx, 'first_four'] = True

    df['predicted_seed'] = df['predicted_seed'].astype(int)

    return df.sort_values(['predicted_seed', 'predicted_seed_raw'])
