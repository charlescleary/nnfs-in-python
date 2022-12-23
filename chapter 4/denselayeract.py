"""_summary_
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Dense Layer
class LayerDense:
    """_summary_"""

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        """_summary_

        Args:
            n_inputs (_type_): _description_
            n_neurons (_type_): _description_
        """
        # Initilalize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_
        """
        # Calculate output values from inputs, weight and biases
        self.outputs = np.dot(inputs, self.weights) + self.biases


# RELU activation
class ActivationRelu:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)


# Softmax activation:
class ActivationSoftmax:

    # Forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 inputs features and 3 output values
dense1 = LayerDense(2, 3)

# Create RELU activation
activation1 = ActivationRelu()

# Create second Dense layer with 3 input features and 3 output values
dense2 = LayerDense(3, 3)

# Create Softmax activation
activation2 = ActivationSoftmax()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
activation1.forward(dense1.outputs)

# Make a forward pass through the second Dense layer
dense2.forward(activation1.output)

# Make a forward pass through the activation function
activation2.forward(dense2.outputs)

# Let's see the output of the first few samples
print(activation2.output[:5])
