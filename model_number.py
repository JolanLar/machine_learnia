import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b
    return 1 / (1 + np.exp(-Z))

def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) + (1 - y) * np.log(1 - A))

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

def artificial_neuron(X, y, learning_rate=0.1, n_iter=100):
    W, b = initialisation(X)
    L = []

    for i in range(n_iter):
        A = model(X, W, b)
        L.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    x0 = np.linspace(min(X[:,0]), max(X[:,0]), 2)
    x1 = (-W[0] * x0 - b) / W[1]
    plt.plot(x0, x1, c='orange', lw=3)
    plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
    plt.show()

    y_pred = predict(X, W, b)
    print(accuracy_score(y, y_pred))

    plt.plot(L)
    plt.show()


# Generate data
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=666)
y = y.reshape((y.shape[0], 1))

# Train the model
artificial_neuron(X, y)