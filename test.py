import numpy as np
from utilities import *
from tqdm import tqdm

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases for the layers
        self.weights_input_hidden = np.random.rand(hidden_size, input_size)
        self.bias_hidden = np.zeros((hidden_size, 1))
        self.weights_hidden_output = np.random.rand(output_size, hidden_size)
        self.bias_output = np.zeros((output_size, 1))

    def forward(self, x):
        # Forward pass through the network
        self.hidden_input = np.dot(self.weights_input_hidden, x) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output = sigmoid(np.dot(self.weights_hidden_output, self.hidden_output) + self.bias_output)
        return self.output

    def backward(self, x, y, learning_rate):
        # Backpropagation to update weights and biases
        error = y - self.output

        d_output = error * sigmoid_derivative(self.output)
        error_hidden = np.dot(self.weights_hidden_output.T, d_output)

        d_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += learning_rate * np.dot(d_output, self.hidden_output.T)
        self.bias_output += learning_rate * d_output
        self.weights_input_hidden += learning_rate * np.dot(d_hidden, x.T)
        self.bias_hidden += learning_rate * d_hidden

    def train(self, X, Y, epochs, learning_rate):
        for _ in tqdm(range(epochs)):
            for x, y in zip(X, Y):
                x = x.reshape(-1, 1)
                self.forward(x)
                self.backward(x, y, learning_rate)

    def predict(self, x):
        x = x.reshape(-1, 1)
        return self.forward(x)

# Example usage:
if __name__ == "__main__":
    # Generate random training data (you should use your own dataset)
    num_samples = 100
    input_size = 64 * 64  # Assuming 64x64 pixel images
    hidden_size = 64
    output_size = 1

    X_train, y_train, X_test, y_test = load_data()

    # Create and train the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X_train, y_train, epochs=100, learning_rate=0.01)

    # Make predictions
    prediction = nn.predict(X_test[0])
    print("Predicted class:", prediction)
    print("Correct class:", y_test[0])