# MNIST Classifiers: Naive Bayes, KNN, and Ensemble Methods

This project applies various classification models to the MNIST dataset to recognize handwritten digits.

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
- Python
- TensorFlow (for dataset)
- scikit-learn
- NumPy
- Matplotlib / Seaborn (optional for visualizations)

## Sample Output
- Accuracy scores
- Confusion matrices
- Classifier comparison

## How to Run
```bash
python naive_bayes_mnist.ipynb
