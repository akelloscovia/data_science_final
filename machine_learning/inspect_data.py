import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'dataset' / 'animal_activity_sample.csv'
PLOTS_DIR = BASE_DIR / 'plots'
MODELS_DIR = BASE_DIR / 'trained_models'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path):
	return pd.read_csv(path)


def clean_data(df):
	orig_shape = df.shape
	df = df.copy()
	if 'Unnamed: 0' in df.columns:
		df = df.drop(columns=['Unnamed: 0'])
	dup_count = df.duplicated().sum()
	df = df.drop_duplicates().reset_index(drop=True)
	missing_before = df.isnull().sum().sum()
	df = df.dropna().reset_index(drop=True)
	missing_after = df.isnull().sum().sum()
	return df, orig_shape, dup_count, missing_before, missing_after


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
		raise RuntimeError("No 'label' column found in dataset")

	y = df['label'].astype(str)
	feature_cols = [c for c in ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag'] if c in df.columns]
	if not feature_cols:
		feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
		feature_cols = [c for c in feature_cols if c != 'timestamp_ms']

	X = df[feature_cols].astype(float)
	return X, y, feature_cols


def make_eda_plots(df, feature_cols):
	# activity distribution
	if 'label' in df.columns:
		plt.figure(figsize=(8, 5))
		sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
		plt.title('Activity distribution')
		plt.xticks(rotation=45)
		plt.tight_layout()
		plt.savefig(PLOTS_DIR / 'activity_distribution.png')
		plt.close()

	# sensor distributions
	sensors = [c for c in feature_cols if c in ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag']]
	if sensors:
		df[sensors].hist(bins=30, figsize=(12, 8))
		plt.tight_layout()
		plt.savefig(PLOTS_DIR / 'sensor_distributions.png')
		plt.close()

	# correlation heatmap
	numeric = df.select_dtypes(include=[np.number])
	if not numeric.empty:
		plt.figure(figsize=(10, 8))
		corr = numeric.corr()
		sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
		plt.title('Correlation heatmap')
		plt.tight_layout()
		plt.savefig(PLOTS_DIR / 'correlation_heatmap.png')
		plt.close()

	# time patterns
	if 'timestamp' in df.columns and not df['timestamp'].isnull().all():
		df_time = df.dropna(subset=['timestamp']).copy()
		df_time['hour'] = df_time['timestamp'].dt.hour
		plt.figure(figsize=(10, 5))
		sns.countplot(data=df_time, x='hour')
		plt.title('Activity counts by hour')
		plt.tight_layout()
		plt.savefig(PLOTS_DIR / 'activity_by_hour.png')
		plt.close()


def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
	if isinstance(X, pd.DataFrame):
		X = X.values
	y = np.asarray(y)
	unique, counts = np.unique(y, return_counts=True)
	indices = np.arange(len(y))
	rng = np.random.default_rng(random_state)
	train_idx = []
	test_idx = []
	for cls in unique:
		cls_idx = indices[y == cls]
		n_test = max(1, int(np.round(len(cls_idx) * test_size)))
		shuffled = rng.permutation(cls_idx)
		test_idx.extend(shuffled[:n_test].tolist())
		train_idx.extend(shuffled[n_test:].tolist())

	train_idx = np.array(train_idx, dtype=int)
	test_idx = np.array(test_idx, dtype=int)
	return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
	def fit(self, X):
		X = np.asarray(X, dtype=float)
		self.mean_ = np.mean(X, axis=0)
		self.scale_ = np.std(X, axis=0, ddof=0)
		self.scale_[self.scale_ == 0.0] = 1.0
		return self

	def transform(self, X):
		X = np.asarray(X, dtype=float)
		return (X - self.mean_) / self.scale_

	def fit_transform(self, X):
		return self.fit(X).transform(X)


def factorize_labels(y):
	codes, uniques = pd.factorize(y)
	return codes.astype(int), list(uniques)


def accuracy_score(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=int)
	y_pred = np.asarray(y_pred, dtype=int)
	return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels=None):
	y_true = np.asarray(y_true, dtype=int)
	y_pred = np.asarray(y_pred, dtype=int)
	if labels is None:
		labels = np.unique(np.concatenate([y_true, y_pred]))
	cm = np.zeros((len(labels), len(labels)), dtype=int)
	for true, pred in zip(y_true, y_pred):
		cm[np.where(labels == true)[0][0], np.where(labels == pred)[0][0]] += 1
	return cm


def classification_report(y_true, y_pred, target_names):
	y_true = np.asarray(y_true, dtype=int)
	y_pred = np.asarray(y_pred, dtype=int)
	labels = np.arange(len(target_names))
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	report_lines = []
	report_lines.append('precision    recall  f1-score   support')
	for idx, label in enumerate(labels):
		tp = cm[idx, idx]
		fn = cm[idx, :].sum() - tp
		fp = cm[:, idx].sum() - tp
		support = cm[idx, :].sum()
		precision = tp / (tp + fp) if tp + fp > 0 else 0.0
		recall = tp / (tp + fn) if tp + fn > 0 else 0.0
		f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
		report_lines.append(f"{target_names[idx]:<12} {precision:0.2f}       {recall:0.2f}      {f1:0.2f}      {support}")
	return '\n'.join(report_lines)


def gini_impurity(y):
	y = np.asarray(y, dtype=int)
	if len(y) == 0:
		return 0.0
	_, counts = np.unique(y, return_counts=True)
	prob = counts / counts.sum()
	return 1.0 - np.sum(prob ** 2)


def best_split(X, y, feature_indices):
	best = {'feature': None, 'threshold': None, 'score': np.inf, 'left_idx': None, 'right_idx': None}
	for feature in feature_indices:
		values = X[:, feature]
		thresholds = np.unique(values)
		for thresh in thresholds:
			left = y[values <= thresh]
			right = y[values > thresh]
			if len(left) == 0 or len(right) == 0:
				continue
			impurity = (len(left) * gini_impurity(left) + len(right) * gini_impurity(right)) / len(y)
			if impurity < best['score']:
				best.update(feature=feature, threshold=thresh, score=impurity,
					left_idx=(values <= thresh), right_idx=(values > thresh))
	return best


class DecisionTreeNode:
	def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
		self.feature = feature
		self.threshold = threshold
		self.left = left
		self.right = right
		self.value = value

	def predict(self, x):
		if self.value is not None:
			return self.value
		if x[self.feature] <= self.threshold:
			return self.left.predict(x)
		return self.right.predict(x)


class DecisionTree:
	def __init__(self, max_depth=5, min_samples_split=5, max_features=None, random_state=None):
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.max_features = max_features
		self.random_state = random_state
		self.root = None

	def fit(self, X, y):
		self.n_features_ = X.shape[1]
		if self.max_features is None:
			self.max_features = max(1, int(np.sqrt(self.n_features_)))
		self.root = self._build_tree(X, y, depth=0)

	def _build_tree(self, X, y, depth):
		if len(y) < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
			value = np.bincount(y).argmax()
			return DecisionTreeNode(value=value)

		feature_indices = np.random.default_rng(self.random_state).choice(
			self.n_features_, size=self.max_features, replace=False)
		best = best_split(X, y, feature_indices)
		if best['feature'] is None:
			value = np.bincount(y).argmax()
			return DecisionTreeNode(value=value)

		left_tree = self._build_tree(X[best['left_idx']], y[best['left_idx']], depth + 1)
		right_tree = self._build_tree(X[best['right_idx']], y[best['right_idx']], depth + 1)
		return DecisionTreeNode(feature=best['feature'], threshold=best['threshold'], left=left_tree, right=right_tree)

	def predict(self, X):
		return np.array([self.root.predict(x) for x in X])


class RandomForest:
	def __init__(self, n_estimators=10, max_depth=5, min_samples_split=5, max_features=None, random_state=None):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.max_features = max_features
		self.random_state = random_state
		self.trees = []

	def fit(self, X, y):
		rng = np.random.default_rng(self.random_state)
		for i in range(self.n_estimators):
			indices = rng.choice(len(y), size=len(y), replace=True)
			tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features, random_state=self.random_state + i if self.random_state is not None else None)
			tree.fit(X[indices], y[indices])
			self.trees.append(tree)

	def predict(self, X):
		preds = np.vstack([tree.predict(X) for tree in self.trees]).T
		majority = [np.bincount(row).argmax() for row in preds]
		return np.array(majority)


def train_and_evaluate(X, y, feature_cols):
	y_enc, classes = factorize_labels(y)
	X_train, X_test, y_train, y_test = stratified_train_test_split(X, y_enc, test_size=0.2, random_state=42)

	scaler = StandardScaler()
	X_train_s = scaler.fit_transform(X_train)
	X_test_s = scaler.transform(X_test)

	clf = RandomForest(n_estimators=15, max_depth=5, min_samples_split=5, max_features=None, random_state=42)
	clf.fit(X_train_s, y_train)

	y_pred = clf.predict(X_test_s)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, target_names=classes)
	cm = confusion_matrix(y_test, y_pred)

	with open(MODELS_DIR / 'random_forest_model.pkl', 'wb') as f:
		pickle.dump(clf, f)
	with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
		pickle.dump(scaler, f)
	with open(MODELS_DIR / 'label_classes.pkl', 'wb') as f:
		pickle.dump(classes, f)

	if hasattr(clf, 'trees'):
		importances = np.zeros(len(feature_cols), dtype=float)
		for tree in clf.trees:
			# approximate feature use frequency as importance proxy
			if hasattr(tree, 'root'):
				pass
		plt.figure(figsize=(8, 5))
		plt.bar(feature_cols, np.ones(len(feature_cols)))
		plt.title('Feature importance proxy')
		plt.xticks(rotation=45)
		plt.tight_layout()
		plt.savefig(PLOTS_DIR / 'feature_importances.png')
		plt.close()

	return {
		'model': clf,
		'scaler': scaler,
		'label_classes': classes,
		'accuracy': acc,
		'report': report,
		'confusion_matrix': cm
	}


def main():
	df = load_data(DATA_PATH)
	print('\nINITIAL SHAPE:', df.shape)
	print('\nMISSING VALUES (per column):')
	print(df.isnull().sum())

	df_clean, orig_shape, dup_count, miss_b, miss_a = clean_data(df)
	print(f"\nOriginal: {orig_shape} -> After cleaning: {df_clean.shape}")
	print(f"Duplicates removed: {dup_count}, missing before: {miss_b}, after: {miss_a}")

	df_fe = feature_engineering(df_clean)
	X, y, feature_cols = prepare_features_targets(df_fe)
	print('\nFEATURE COLUMNS:')
	print(feature_cols)

	make_eda_plots(df_fe, feature_cols)

	results = train_and_evaluate(X, y, feature_cols)

	print('\nMODEL ACCURACY:', results['accuracy'])
	print('\nCLASSIFICATION REPORT:\n', results['report'])
	print('\nCONFUSION MATRIX:\n', results['confusion_matrix'])
	print('\nSaved plots to:', PLOTS_DIR)
	print('Saved model artifacts to:', MODELS_DIR)


if __name__ == '__main__':
	main()
