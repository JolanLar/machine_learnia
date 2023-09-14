import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utilities import *
from tqdm import tqdm

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b
    return 1 / (1 + np.exp(-Z))

def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)

def update(dW, db, W, b, learning_rate):
    W -= dW * learning_rate
    b -= db * learning_rate
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5

def artificial_neuron(X_train, y_train, X_test, y_test, learning_rate=0.01, n_iter=1000):
    W, b = initialisation(X_train)
    L_train = []
    acc_train = []
    L_test = []
    acc_test = []

    for i in tqdm(range(n_iter)):
        A = model(X_train, W, b)

        if i % 10 == 0:
            L_train.append(log_loss(y_train, A))
            y_pred = predict(X_train, W, b)
            acc_train.append(accuracy_score(y_train, y_pred))
        
            A_test = model(X_test, W, b)
            L_test.append(log_loss(y_test, A_test))
            y_pred = predict(X_test, W, b)
            acc_test.append(accuracy_score(y_test, y_pred))
        
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(L_train, label='train')
    plt.plot(L_test, label='test')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(acc_train, label='train')
    plt.plot(acc_test, label='test')
    plt.legend()
    plt.show()

    return (W, b)


# Get datas
X_train, y_train, X_test, y_test = load_data()
X_max = np.max(X_train[:,0])
X_train = X_train.reshape(X_train.shape[0], -1) / X_max
X_test = X_test.reshape(X_test.shape[0], -1) / X_max


# Train the model
W, b = artificial_neuron(X_train, y_train, X_test, y_test, n_iter=10000, learning_rate=0.01)