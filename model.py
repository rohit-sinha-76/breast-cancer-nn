import numpy as np
from utils import sigmoid, relu

def initialize_parameters(n_x, n_h, n_y):
    """
    Initializes parameters for a 2-layer neural network.
    
    Args:
        n_x (int): Size of the input layer.
        n_h (int): Size of the hidden layer.
        n_y (int): Size of the output layer.
        
    Returns:
        dict: Initialized weights and biases.
    """
    parameters = {
        "W1": np.random.randn(n_h, n_x) * 0.01,
        "b1": np.zeros((n_h, 1)),
        "W2": np.random.randn(n_y, n_h) * 0.01,
        "b2": np.zeros((n_y, 1))
    }
    return parameters

def forward_propagation(X, parameters):
    """
    Performs forward propagation.
    
    Args:
        X (np.ndarray): Input data.
        parameters (dict): Network parameters.
        
    Returns:
        tuple: (cache dictionary used in backpropagation, output A2)
    """
    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
    A1 = relu(Z1)
    
    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return cache, A2

def backward_propagation(parameters, cache, X, y):
    """
    Computes gradients using backpropagation.
    
    Returns:
        dict: Gradients computed.
    """
    m = X.shape[1]
    
    W2 = parameters["W2"]
    A1 = cache["A1"]
    Z1 = cache["Z1"]
    A2 = cache["A2"]

    # Output layer derivatives
    dZ2 = A2 - y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer derivatives
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2
    }
    return grads

def update_parameters(parameters, grads, learning_rate):
    """Updates the neural network parameters using gradient descent."""
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]

    return parameters

def predict(X, parameters):
    """Predicts binary labels for a given set of input features."""
    _, A2 = forward_propagation(X, parameters)
    predictions = (A2 >= 0.5).astype(int)
    return predictions
