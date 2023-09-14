import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utilities import *
from tqdm import tqdm


class layer:
    def __init__(self, n_input, n_output):
        self.weights = np.random.randn(n_output, n_input)
        self.biais = np.random.randn(n_output, 1)

class neural_network:
    def __init__(self, size=[3, 32, 3], learning_rate=0.01):
        self.__learning_rate = learning_rate
        self.__layers_result = []
        self.__layers = []
        for i in range(1, len(size)):
            self.__layers.append(layer(size[i-1], size[i]))


    def __forward_propagation(self, inputs):
        self.__layers_result = []

        for i_layer in range(len(self.__layers)):
            inputs = 1 / ( 1 + np.exp(-(self.__layers[i_layer].weights.dot(inputs) + self.__layers[i_layer].biais)))
            self.__layers_result.append(inputs)

        return inputs


    def __log_loss(self, wanted_result, current_result):
        epsilon = 1e-15
        return 1 / len(current_result) * np.sum(-current_result * np.log(wanted_result + epsilon) - (1 - current_result) * np.log(1 - wanted_result + epsilon))


    def __backward_propagation(self, inputs, outputs):

        dividend = 1 / outputs.shape[1]
        n_layers = len(self.__layers_result)

        dZ = self.__layers_result[-1] - outputs

        if n_layers == 1:
            dW = dividend * dZ.dot(inputs.T)
            db = dividend * np.sum(dZ, axis=1, keepdims=2)

            self.__layers[0].weights -= dW * self.__learning_rate
            self.__layers[0].biais -= db * self.__learning_rate
            

        else:
            dW = dividend * dZ.dot(self.__layers_result[-2].T)
            db = dividend * np.sum(dZ, axis=1, keepdims=2)

            for i_layer in reversed(range(1, n_layers - 1)):
                i_next_layer = i_layer + 1
                i_previous_layer = i_layer - 1

                dZ = self.__layers[i_next_layer].weights.T.dot(dZ) * self.__layers_result[i_layer] * (1 - self.__layers_result[i_layer])

                self.__layers[i_next_layer].weights -= dW * self.__learning_rate
                self.__layers[i_next_layer].biais -= db * self.__learning_rate

                dW = dividend * dZ.dot(self.__layers_result[i_previous_layer].T)
                db = dividend * np.sum(dZ, axis=1, keepdims=2)
            
            dZ = self.__layers[1].weights.T.dot(dZ) * self.__layers_result[0] * (1 - self.__layers_result[0])

            self.__layers[1].weights -= dW * self.__learning_rate
            self.__layers[1].biais -= db * self.__learning_rate
            
            dW = dividend * dZ.dot(inputs.T)
            db = dividend * np.sum(dZ, axis=1, keepdims=2)

            self.__layers[0].weights -= dW * self.__learning_rate
            self.__layers[0].biais -= db * self.__learning_rate


    def predict(self, inputs):
        result = self.__forward_propagation(inputs)
        return result >= 0.5

    def train(self, X_train, y_train, n_iter=1000, X_test=[], y_test=[]):

        L_train = []
        acc_train = []

        L_test = []
        acc_test = []

        for i in tqdm(range(n_iter)):
            result = self.__forward_propagation(X_train)
            self.__backward_propagation(X_train, y_train)

            if i % 20 == 0:
                L_train.append(self.__log_loss(result, y_train))
                y_pred = self.predict(X_train)
                acc_train.append(accuracy_score(y_train.flatten(), y_pred.flatten()))
            
                if len(X_test) != 0:
                    test_result = self.__forward_propagation(X_test)
                    L_test.append(self.__log_loss(test_result, y_test))
                    y_pred = self.predict(X_test)
                    acc_test.append(accuracy_score(y_test.flatten(), y_pred.flatten()))
            

        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 2)
        x0 = np.linspace(min(X_train.T[:,0]), max(X_train.T[:,0]), 2)
        x1 = (-self.__layers[-1].weights[0][0] * x0 - self.__layers[-1].biais[0]) / self.__layers[-1].weights[0][1]
        plt.plot(x0, x1.T, c='orange', lw=3)
        plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
        plt.subplot(2, 2, 1)
        plt.plot(L_train, label='train')
        plt.plot(L_test, label='test')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(acc_train, label='train')
        plt.plot(acc_test, label='test')
        plt.legend()
        plt.show()
        

# Get datas
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=12341324)
y = y.reshape((y.shape[0], 1))

# Train the model
nn = neural_network(size=[2, 32, 1])
nn.train(X.T, y.T, n_iter=10000)