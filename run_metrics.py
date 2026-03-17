import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append(r'c:\Users\rohit\OneDrive\Desktop\deep-\breast-cancer-nn')
from data import load_and_prepare_data
from train import train_model
from model import predict
from utils import accuracy
import numpy as np

# Load
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Train
parameters, costs = train_model(X_train, y_train, n_h=4, learning_rate=0.001, num_iterations=1000)

# Predict
train_preds = predict(X_train, parameters)
test_preds = predict(X_test, parameters)

# Metric
train_acc = accuracy(train_preds, y_train)
test_acc = accuracy(test_preds, y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.flatten(), test_preds.flatten())

# Save
with open(r'c:\Users\rohit\OneDrive\Desktop\deep-\breast-cancer-nn\results.txt', 'w') as f:
    f.write(f"Train Accuracy: {train_acc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")
