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


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 inputs features and 3 output values
dense1 = LayerDense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see the output of the first few samples
print(dense1.outputs[:5])
