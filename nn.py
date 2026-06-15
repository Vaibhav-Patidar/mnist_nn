import numpy as np

def init_params():
    layers = [784, 128, 64, 10]
    params = {}
    for i in range(len(layers) - 1):
        params[f'W{i+1}'] = np.random.randn(layers[i], layers[i+1]) * 0.01
        params[f'b{i+1}'] = np.zeros((1, layers[i+1]))
    return params

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)
     

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward(X, params):
    cache = {'A0': X}
    L = len(params) // 2

# yha pr ye kra gaya hai for layers like 128, 64
    for i in range (1, L):
        Z = cache[f'A{i-1}'] @ params[f'W{i}'] + params[f'b{i}']
        cache[f'Z{i}'] = Z
        cache[f'A{i}'] = relu(Z)

# yeh sirf last layer ke liye kara hai
    Z = cache[f'A{L-1}'] @ params[f'W{L}'] + params[f'b{L}']
    cache[f'Z{L}'] = Z
    cache[f'A{L}'] = softmax(Z)

    return cache[f'A{L}'], cache

def compute_loss(y_pred, y_true):
    m = y_true.shape[0]
    correct_class_probs = y_pred[np.arange(m), np.argmax(y_true, axis=1)]
    loss = -np.mean(np.log(correct_class_probs + 1e-8))
    return loss

def backward(params, cache, y_true):
    L = len(params) // 2
    m = y_true.shape[0]
    grads = {}
    dZ = cache[f'A{L}'] - y_true
    for i in range (L,0, -1):
        grads[f'dW{i}'] = (cache[f'A{i-1}']).T @ dZ / m
        grads[f'db{i}'] = np.mean(dZ, axis=0, keepdims=True)
        if i > 1:
            dZ = dZ @ params[f'W{i}'].T * relu_derivative(cache[f'Z{i-1}'])
    return grads

def update_params(grads,params,learning_rate):
    L = len(params) // 2
    for i in range(1, L+1):
        params[f'W{i}'] = params[f'W{i}'] - learning_rate * grads[f'dW{i}']
        params[f'b{i}'] = params[f'b{i}'] - learning_rate * grads[f'db{i}']
    return params

