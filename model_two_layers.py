import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utilities import *
from tqdm import tqdm

def initialisation(n0, n1, n2):
    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)

    return (W1, b1, W2, b2)


def forward_propagation(X, W1, b1, W2, b2):

    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))

    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    return (A1, A2)


def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


def back_propagation(X, y, A1, A2, W2):

    dividend = 1 / y.shape[1]

    dZ2 = A2 - y
    dW2 = dividend * dZ2.dot(A1.T)
    db2 = dividend * np.sum(dZ2, axis=1, keepdims=2)

    dZ1 = W2.T.dot(dZ2) * A1 * (1 - A1)
    dW1 = dividend * dZ1.dot(X.T)
    db1 = dividend * np.sum(dZ1, axis=1, keepdims=2)

    return  dW1, db1, dW2, db2


def update(dW1, db1, W1, b1, dW2, db2, W2, b2, learning_rate):
    W1 -= dW1 * learning_rate
    b1 -= db1 * learning_rate
    W2 -= dW2 * learning_rate
    b2 -= db2 * learning_rate
    return (W1, b1, W2, b2)


def predict(X, W1, b1, W2, b2):
    _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return A2 >= 0.5


def neurol_network(X_train, y_train, X_test, y_test, n1, learning_rate=0.01, n_iter=1000):

    W1, b1, W2, b2 = initialisation(X_train.shape[0], n1, y_train.shape[0])

    L_train = []
    acc_train = []
    L_test = []
    acc_test = []

    for i in tqdm(range(n_iter)):
        A1, A2 = forward_propagation(X_train, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_propagation(X_train, y_train, A1, A2, W2)
        W1, b1, W2, b2 = update(dW1, db1, W1, b1, dW2, db2, W2, b2, learning_rate)

        if i % 30 == 0:
            L_train.append(log_loss(y_train, A2))
            y_pred = predict(X_train, W1, b1, W2, b2)
            acc_train.append(accuracy_score(y_train.flatten(), y_pred.flatten()))
        
            _, A_test = forward_propagation(X_test, W1, b1, W2, b2)
            L_test.append(log_loss(y_test, A_test))
            y_pred = predict(X_test, W1, b1, W2, b2)
            acc_test.append(accuracy_score(y_test.flatten(), y_pred.flatten()))
        

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
X_train = (X_train.reshape(X_train.shape[0], -1) / X_max).T
y_train = y_train.T
X_test = (X_test.reshape(X_test.shape[0], -1) / X_max).T
y_test = y_test.T

# Train the model
W, b = neurol_network(X_train, y_train, X_test, y_test, 32, n_iter=50000, learning_rate=0.01)