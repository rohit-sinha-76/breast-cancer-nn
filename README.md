# Breast Cancer Classification: 2-Layer Neural Network (From-Scratch)

**Author**: Rohit  
**Focus**: Machine Learning Math & Linear Algebra (From-Scratch Backpropagation)  
**Tech Stack**: Python, NumPy, Scikit-learn (Data Prep only), Matplotlib  

---

## Why This Project Highlights Core ML Rigor
While modern frameworks like TensorFlow or PyTorch abstract away the complexity, this repository contains a **ground-up implementation** of a 2-layer Neural Network designed with **only NumPy**. 

By manually implementing Backpropagation, Cost computation with Cross-Entropy, and Adaptive weight updating, it demonstrates a complete mastery of the math underpinning Deep Learning (Linear Algebra and Chain Rule Calculus).

---

## Key Results & Evaluation
The model was evaluated on the **Wisconsin Breast Cancer Dataset** (569 cases, 30 continuous features). 

| Metric | Score (%) |
| :--- | :--- |
| **Training Accuracy** | **91.21%** |
| **Testing Accuracy** (Unseen Data) | **93.86%** |

### Confusion Matrix Breakdown (Test Set)
In medical applications, **Recall (Sensitivity)** for the Malignant class is crucial (failing to detect cancer carries high risk).

Columns: `[Malignant, Benign]`

```text
[ [38,  5]   <-- Malignant (True Malignant: 38, Missed Malignant: 5)
  [ 2, 69] ]  <-- Benign (False Malignant: 2, True Benign: 69)
```

**Malignant Class Metrics**:
*   **Precision**: 95.00% (When predicts malignant, it is malignant)
*   **Recall**: **88.37%** (Identified 38 out of 43 total malignant cases)

---

## Model Architecture & Dimensions
The network is designed with a **Separation of Concerns (SoC)** approach, making it modular and easy to extend.

| Layer | Type | Nodes | Input Shape | Output Shape | Activation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Features | 30 | `(30, m)` | `(30, m)` | N/A |
| **Layer 1** | Hidden | 4 | `(30, m)` | `(4, m)` | **ReLU** |
| **Layer 2** | Output | 1 | `(4, m)` | `(1, m)` | **Sigmoid** |

*   **Cost Function**: Cross-Entropy Loss (Log Loss) with stability epsilon `1e-8`.
*   **Optimizer**: Batch Gradient Descent.

---

## Project Structure
*   `data.py`: Connects to `load_breast_cancer()` and prepares standard vector dimensions (Matrices transposed).
*   `model.py`: Matrix forward and backward prop implementation with Sigmoid/ReLU activations.
*   `train.py`: Iterative gradient descent loop for weights $(W)$ and bias $(b)$ updates.
*   `utils.py`: Contains cost metrics and activation handlers.
*   `main.py`: Full execution script delivering console metrics and decision-boundary curves.

---

## How to Run & Reproduce
1.  **Clone inside the workspace directory**
2.  **Install requirements**:
    ```bash
    pip install numpy scikit-learn matplotlib
    ```
3.  **Execute Training & Plot Graphics**:
    ```bash
    python main.py
    ```
    *Will print accuracies automatically and prompt decision-curve distribution histograms.*
