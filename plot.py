import numpy as np
import matplotlib.pyplot as plt
import csv
from train import train
from evaluate import evaluate

def load_history(path='training_history.csv'):
    epochs, losses, accuracies = [], [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['loss']))
            accuracies.append(float(row['accuracy']))
    return epochs, losses, accuracies

def plot_loss_accuracy(epochs, losses, accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MNIST Neural Network Training — NumPy from Scratch', fontsize=14, fontweight='bold')

    ax1.plot(epochs, losses, color='#e74c3c', linewidth=2)
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a*100 for a in accuracies], color='#2ecc71', linewidth=2)
    ax2.set_title('Training Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: training_curves.png")

def plot_confusion_matrix(params):
    from data import load_mnist
    from nn import forward

    X_train, y_train, X_test, y_test, y_test_raw = load_mnist()
    y_pred, _ = forward(X_test, params)
    predictions = np.argmax(y_pred, axis=1)
    actual = np.argmax(y_test, axis=1)

    cm = np.zeros((10, 10), dtype=int)
    for p, a in zip(predictions, actual):
        cm[a][p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix — Test Set')

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                    color='white' if cm[i][j] > cm.max()//2 else 'black', fontsize=8)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: confusion_matrix.png")

if __name__ == "__main__":
    epochs, losses, accuracies = load_history()
    plot_loss_accuracy(epochs, losses, accuracies)
    params = np.load('params.npy', allow_pickle=True).item()
    plot_confusion_matrix(params)