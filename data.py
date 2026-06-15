import numpy as np
import matplotlib.pyplot as plt

def load_mnist():
    import urllib.request
    import gzip
    import os

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }

    os.makedirs("mnist_data", exist_ok=True)

    for key, filename in files.items():
        path = f"mnist_data/{filename}"
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, path)

    def read_images(path):
        with gzip.open(path, 'rb') as f:
            f.read(16)  # skip header
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(-1, 784) / 255.0

    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            f.read(8)  # skip header
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = read_images("mnist_data/train-images-idx3-ubyte.gz")
    y_train = read_labels("mnist_data/train-labels-idx1-ubyte.gz")
    X_test  = read_images("mnist_data/t10k-images-idx3-ubyte.gz")
    y_test  = read_labels("mnist_data/t10k-labels-idx1-ubyte.gz")

    def one_hot(y, classes=10):
        oh = np.zeros((y.shape[0], classes))
        oh[np.arange(y.shape[0]), y] = 1
        return oh

    return X_train, one_hot(y_train), X_test, one_hot(y_test), y_test

def preview(X, y_raw, index=0):
    plt.imshow(X[index].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_raw[index]}")
    plt.show()

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, y_test_raw = load_mnist()
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    preview(X_train, y_test_raw)