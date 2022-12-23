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


# Common loss class
class Loss:

    # Calculate the data and regularization losses
    # given model output and ground truth vlaues
    def calculate(self, output, y):

        # calculate sample loss
        sample_losses = self.forward(output, y)

        # calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss


# Cross-entropy loss
class LossCategoricalCrossEntropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both side to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target values
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelyhoods = -np.log(correct_confidences)
        return negative_log_likelyhoods


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

# Create loss function
loss_function = LossCategoricalCrossEntropy()

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

# perform a forward pass through the loss function
# it takes the output of the second dense layer and return loss
loss = loss_function.calculate(activation2.output, y)

# print the loss value
print("loss: ", loss)

# Calculate the accuracy from the output of activation2 and targets
# calculate values along the first axis
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print("acc: ", accuracy)
