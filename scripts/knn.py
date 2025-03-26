import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten images
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Try KNN with different k values
for k in [2, 3, 4, 5]:
    print(f"\n=== KNN Classifier (k = {k}) ===")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))