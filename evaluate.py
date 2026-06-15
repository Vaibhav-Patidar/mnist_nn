import numpy as np
from data import load_mnist
from nn import init_params, forward

def evaluate(params):
    X_train, y_train, X_test, y_test, y_test_raw = load_mnist()
    correct = 0
    total = 0
    y_pred, cache = forward(X_test, params)
    predictions = np.argmax(y_pred, axis=1)
    actual = np.argmax(y_test, axis = 1)
  
    accuracy = np.mean(predictions == actual)

    print(f"Accuracy: {accuracy}")