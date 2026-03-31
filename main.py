"""
Main entry point for training and evaluating the Breast Cancer Neural Network.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from data import load_and_prepare_data
from train import train_model
from model import predict, forward_propagation
from utils import accuracy

def main():
    print("Loading and preparing dataset...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape:  {X_test.shape}")
    
    print("\nStarting model training...")
    # Train the neural network
    parameters, costs = train_model(
        X_train, y_train, 
        n_h=4, 
        learning_rate=0.001, 
        num_iterations=1000
    )
    
    print("\nEvaluating model performance...")
    # Measure predictions
    train_preds = predict(X_train, parameters)
    test_preds = predict(X_test, parameters)
    
    # Calculate and display metrics
    train_acc = accuracy(train_preds, y_train)
    test_acc = accuracy(test_preds, y_test)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    
    # Confusion matrix representation
    cm = confusion_matrix(y_test.flatten(), test_preds.flatten())
    print("\nConfusion Matrix (Test Data):")
    print(cm)
    
    # Visualize training progress and prediction confidence
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Cost iteration curve
    plt.subplot(1, 2, 1)
    plt.plot(costs, color='blue')
    plt.xlabel("Iterations (x10)")
    plt.ylabel("Cost")
    plt.title("Training Cost via Gradient Descent")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Subplot 2: Confidence distribution histogram
    _, A2_train = forward_propagation(X_train, parameters)
    plt.subplot(1, 2, 2)
    plt.hist(A2_train.flatten(), bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency Count")
    plt.title("Prediction Confidence Distribution (Train)")
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    import os
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/breast_cancer_training_results.png')
    print("Saved visualization to plots/")

if __name__ == "__main__":
    main()
