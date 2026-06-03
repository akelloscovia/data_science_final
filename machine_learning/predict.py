import pickle
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'trained_models'
DATA_PATH = BASE_DIR / 'dataset' / 'animal_activity_sample.csv'
MODEL_PATH = MODELS_DIR / 'random_forest_model.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
LABELS_PATH = MODELS_DIR / 'label_classes.pkl'
FEATURE_COLUMNS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag']


def build_random_forest_model():
    """Train the random forest model and persist artifacts."""
    if not DATA_PATH.exists():
        raise FileNotFoundError('Training dataset is missing.')

    try:
        from machine_learning.training import (
            clean_data,
            feature_engineering,
            load_data,
            prepare_features_targets,
            train_random_forest,
        )
    except ImportError as exc:
        raise ImportError(
            'Unable to import training dependencies. Ensure scikit-learn is installed and not blocked by policy.'
        ) from exc

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = feature_engineering(df)
    X, y, _ = prepare_features_targets(df)
    train_random_forest(X, y)


def load_random_forest_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not LABELS_PATH.exists():
        build_random_forest_model()

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LABELS_PATH, 'rb') as f:
        label_classes = pickle.load(f)

    return {
        'model': model,
        'scaler': scaler,
        'label_classes': label_classes,
    }


def predict(features):
    artifact = load_random_forest_model()
    if artifact is None:
        raise FileNotFoundError('Trained model artifacts are not available.')

    model = artifact['model']
    scaler = artifact['scaler']
    label_classes = artifact['label_classes']

    if isinstance(features, pd.DataFrame):
        X = features
    else:
        X = np.asarray(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] == len(FEATURE_COLUMNS):
            X = pd.DataFrame(X, columns=FEATURE_COLUMNS)

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return [label_classes[int(pred)] for pred in predictions]
