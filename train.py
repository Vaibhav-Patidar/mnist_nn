import numpy as np
from data import load_mnist
from nn import init_params, forward, compute_loss, backward, update_params
from evaluate import evaluate

def train(epochs=1500, learning_rate=0.1):
    X_train, y_train, X_test, y_test, y_test_raw = load_mnist()
    params = init_params()
    history = {'epoch': [], 'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        y_pred, cache = forward(X_train, params)
        y_true = y_train 
        
        loss = compute_loss(y_pred, y_true)
        grads = backward(params,cache,y_true)
        params = update_params(grads,params,learning_rate)
        
        if epoch % 10 == 0:
            predictions = np.argmax(y_pred, axis=1)
            actual = np.argmax(y_true, axis=1)
            acc = np.mean(predictions == actual)
            
            history['epoch'].append(epoch)
            history['loss'].append(loss)
            history['accuracy'].append(acc)
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

    import csv
    with open('training_history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'accuracy'])
        writer.writeheader()
        for i in range(len(history['epoch'])):
            writer.writerow({
                'epoch': history['epoch'][i],
                'loss': history['loss'][i],
                'accuracy': history['accuracy'][i]
            })
    np.save('params.npy', params)
    return params, history

if __name__ == "__main__":
    params, history = train()
    evaluate(params)