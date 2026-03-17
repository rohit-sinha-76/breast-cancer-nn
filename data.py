from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_and_prepare_data(test_size=0.2, random_state=42):
    """
    Loads the breast cancer dataset and splits it into training and testing sets.
    Reshapes the data to be compatible with the neural network dimensions 
    (features as rows, examples as columns).
    
    Args:
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Transpose matrices to treat each column as a separate training example
    X_train = X_train.T          
    X_test = X_test.T            
    
    # Reshape labels to standard vector dimensions 
    y_train = y_train.reshape(1, -1)  
    y_test = y_test.reshape(1, -1)    
    
    return X_train, X_test, y_train, y_test