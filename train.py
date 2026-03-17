from model import (
    initialize_parameters, 
    forward_propagation, 
    backward_propagation, 
    update_parameters
)
from utils import compute_cost

def train_model(X_train, y_train, n_h=4, learning_rate=0.001, num_iterations=1000):
    """
    Trains a 2-layer neural network model.
    
    Args:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): True labels.
        n_h (int): Number of nodes in the hidden layer.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for optimization loops.
        
    Returns:
        tuple: (optimized parameters, list of cost values per 10 iterations)
    """
    n_x = X_train.shape[0]
    n_y = y_train.shape[0]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []

    for i in range(num_iterations + 1):
        # Forward propagation
        cache, A2 = forward_propagation(X_train, parameters)
        
        # Cost computation
        cost = compute_cost(A2, y_train)
        if i % 10 == 0:
            costs.append(cost)
            
        # Backward propagation
        grads = backward_propagation(parameters, cache, X_train, y_train)
        
        # Gradient descent update
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Logging progress
        if i % 100 == 0:
            print(f"Iteration {i:4d} | Cost: {cost:.4f}")

    return parameters, costs