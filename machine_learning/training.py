import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'dataset' / 'animal_activity_sample.csv'
MODELS_DIR = BASE_DIR / 'trained_models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df


def feature_engineering(df):
    df = df.copy()
    if 'timestamp_ms' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')
        if df['timestamp'].isnull().all():
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='s', errors='coerce')
    else:
        df['timestamp'] = pd.NaT

    for axes, name in ((['ax', 'ay', 'az'], 'acc'), (['gx', 'gy', 'gz'], 'gyro')):
        if all(col in df.columns for col in axes):
            df[f'{name}_mag'] = np.sqrt((df[axes] ** 2).sum(axis=1))

    return df


def prepare_features_targets(df):
    if 'label' not in df.columns:
        raise RuntimeError('No label column found.')
    y = df['label'].astype(str)
    feature_cols = [c for c in ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag'] if c in df.columns]
    if not feature_cols:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c != 'timestamp_ms']
    X = df[feature_cols].astype(float)
    return X, y, feature_cols


def train_random_forest(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    with open(MODELS_DIR / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / 'label_classes.pkl', 'wb') as f:
        pickle.dump(list(le.classes_), f)

    return clf, scaler, le


def create_sequence_data(X, y, window_size=20, step=10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    sequences = []
    labels = []
    for start in range(0, len(X) - window_size + 1, step):
        end = start + window_size
        sequences.append(X[start:end])
        labels.append(y[end - 1])
    return np.array(sequences), np.array(labels)


def train_lstm(X, y, feature_cols, epochs=15, batch_size=32):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as exc:
        print('TensorFlow is unavailable for LSTM training:', exc)
        return None

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_sequences, y_sequences = create_sequence_data(X, y_enc, window_size=20, step=10)
    if len(X_sequences) == 0:
        raise RuntimeError('Not enough data to build sequence windows for LSTM training.')

    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences)
    scaler = StandardScaler()
    nsamples, ntimesteps, nfeatures = X_train.shape
    X_train_flat = scaler.fit_transform(X_train.reshape(nsamples, nfeatures * ntimesteps))
    X_train = X_train_flat.reshape(nsamples, ntimesteps, nfeatures)

    model = keras.Sequential([
        keras.layers.Input(shape=(ntimesteps, nfeatures)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(len(np.unique(y_enc)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    model.save(MODELS_DIR / 'lstm_model')
    return model


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = feature_engineering(df)
    X, y, feature_cols = prepare_features_targets(df)

    print('Training Random Forest model...')
    train_random_forest(X, y)
    print('Random Forest training complete.')

    print('Attempting LSTM training...')
    train_lstm(X, y, feature_cols)
    print('Training pipeline finished.')


if __name__ == '__main__':
    main()
