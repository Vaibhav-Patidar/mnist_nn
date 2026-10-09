import json
import numpy as np
from data import load_mnist
from nn import forward

params = np.load("params.npy", allow_pickle=True).item()
L = len(params) // 2
acts = ["relu"] * (L - 1) + ["softmax"]

# round for a smaller file
rounded = {}
layers = []
for i in range(1, L + 1):
    W = np.round(params[f"W{i}"], 4)
    b = np.round(params[f"b{i}"], 4)
    rounded[f"W{i}"], rounded[f"b{i}"] = W, b
    layers.append({"W": W.tolist(), "b": b.ravel().tolist(), "act": acts[i - 1]})

# check that rounding didn't hurt accuracy
_, _, X_test, _, y_test_raw = load_mnist()
probs, _ = forward(X_test, rounded)
acc = float(np.mean(np.argmax(probs, axis=1) == y_test_raw))
print(f"Test accuracy with exported weights: {acc*100:.2f}%")

json.dump({"input": 784, "test_accuracy": round(acc, 4), "layers": layers},
          open("weights.json", "w"))

# test vectors so the JS implementation can be checked against NumPy
samples = []
for idx in range(8):
    p, _ = forward(X_test[idx:idx+1], rounded)
    samples.append({"x": np.round(X_test[idx], 3).tolist(),
                    "label": int(y_test_raw[idx]),
                    "probs": np.round(p[0], 5).tolist()})
json.dump({"samples": samples}, open("test_vectors.json", "w"))
print("Wrote weights.json and test_vectors.json")