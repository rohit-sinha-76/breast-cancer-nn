import numpy as np

def sigmoid(Z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    """ReLU activation function."""
    return np.maximum(0, Z)

def compute_cost(A2, y):
    """
    Computes the cross-entropy cost.
    
    Args:
        A2 (np.ndarray): Post-activation output of the second layer, shape (1, number of examples).
        y (np.ndarray): True label vector, shape (1, number of examples).
        
    Returns:
        float: Cross-entropy cost.
    """
    m = y.shape[1]
    eps = 1e-8
    
    cost = -(1 / m) * np.sum(
        y * np.log(A2 + eps) + (1 - y) * np.log(1 - A2 + eps)
    )
    return cost

def accuracy(predictions, y):
    """Computes the accuracy of predictions against true labels."""
    return np.mean(predictions == y)
