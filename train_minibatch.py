import numpy as np
from data import load_mnist
from nn import init_params, forward, backward, update_params
from evaluate import evaluate

def train_minibatch(epochs=30, lr=0.1, batch=128, seed=0):
    np.random.seed(seed)
    X, Y, Xt, Yt, yt_raw = load_mnist()
    params = init_params()

    # He initialisation (the 0.01 init learns very slowly)
    for i in range(1, len(params) // 2 + 1):
        fan_in = params[f"W{i}"].shape[0]
        params[f"W{i}"] = np.random.randn(*params[f"W{i}"].shape) * np.sqrt(2.0 / fan_in)

    n = X.shape[0]
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            _, cache = forward(X[idx], params)
            grads = backward(params, cache, Y[idx])
            params = update_params(grads, params, lr)
        probs, _ = forward(Xt, params)
        acc = np.mean(np.argmax(probs, axis=1) == yt_raw)
        print(f"Epoch {epoch+1}/{epochs} | Test accuracy: {acc*100:.2f}%")

    np.save("params.npy", params)
    return params

if __name__ == "__main__":
    params = train_minibatch()
    evaluate(params)