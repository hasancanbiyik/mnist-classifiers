# MNIST Classifiers (with Traditional ML Methods)

A collection of machine‑learning pipelines demonstrating classical classification algorithms on the MNIST handwritten‑digits dataset. Each approach includes end‑to‑end code for data loading, preprocessing, model training, evaluation, and performance comparison.

## Classifiers Implemented
- Naive Bayes (Gaussian & Multinomial NB)
- K-Nearest Neighbors (KNN, k=2, 3, 4, 5)
- Decision Tree
- Support Vector Machine (SVM)
- Ensemble Models (Bagging, Random Forest, AdaBoost, XGBoost)

## Dataset
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/) via `tensorflow.keras.datasets`

## Preprocessing
- Normalized pixel values between 0 and 1
- Data reshaped for model compatibility

## Technologies Used
Python • TensorFlow • scikit‑learn • NumPy • Matplotlib • Seaborn • XGBoost  

## Sample Output
- Accuracy scores
- Confusion matrices
- Classifier comparison

## How to Run
```bash
git clone https://github.com/<your-username>/mnist-project.git
cd mnist-project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage
``` python scripts/naive_bayes.py
python scripts/knn.py
python scripts/decision_tree.py
python scripts/ensemble.py
python scripts/svm.py
```


This version is **clear**, **complete**, and **ready for recruiters** to clone, install, and run in seconds.
