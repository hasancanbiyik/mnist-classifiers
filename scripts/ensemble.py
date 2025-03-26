import tensorflow as tf
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten images
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Bagging
bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, n_jobs=-1, random_state=42)
bag.fit(x_train, y_train)
y_pred = bag.predict(x_test)
print("=== Bagging ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# AdaBoost
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42)
ada.fit(x_train, y_train)
y_pred = ada.predict(x_test)
print("\n=== AdaBoost ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# XGBoost
xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="mlogloss", random_state=42)
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
print("\n=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))