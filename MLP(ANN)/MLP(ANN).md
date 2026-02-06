## 🧠 Multilayer Perceptron (MLP) - Complete Guide
#### What is an MLP?
MLP = Multi-Layer PerceptronIt is a type of Artificial Neural Network (ANN)
"Multi-layer" → has one or more hidden layers between input and output
Each layer contains neurons (also called nodes or units)
Neurons perform two operations:

Take weighted sum of inputs + bias → Z
Apply activation function → A
#### 2️⃣ Structure of an MLP
Input layer → Hidden layer(s) → Output layer
Layer Breakdown:
Input layer: Your features/input data

Hidden layer(s): Neurons that learn patterns and representations

Output layer: Final prediction/classification

#### Example Architecture:
Input: 4 features → n₀ = 4
Hidden layer: 3 neurons → n₁ = 3
Output: 2 classes → n₂ = 2
#### 3️⃣ Forward Propagation (Step by Step)
Step 1: Linear Combination
math
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
Where:

A^{[0]} = X (input data)

W^{[l]} shape = (neurons in current layer, neurons in previous layer)

b^{[l]} shape = (neurons in current layer, 1)

Step 2: Activation
math
A^{[l]} = f(Z^{[l]})
Where f is a non-linear activation function (ReLU, Sigmoid, Tanh, etc.)

Step 3: Repeat
Repeat Steps 1-2 for each hidden layer → output layer

#### 4️⃣ Shapes Recap
Layer	Input Shape	Weight Shape	Bias Shape	Z Shape	Activation Shape
Hidden1	(n₀, m)	(n₁, n₀)	(n₁, 1)	(n₁, m)	(n₁, m)
Hidden2	(n₁, m)	(n₂, n₁)	(n₂, 1)	(n₂, m)	(n₂, m)
Output	(n₂, m)	(n_y, n₂)	(n_y, 1)	(n_y, m)	(n_y, m)
Where:

n₀ = input features

n₁, n₂ = hidden layer neurons

n_y = output neurons

m = batch size

#### 5️⃣ Activation Functions (Basics)
Function	Formula	Common Use
ReLU	f(z) = max(0, z)	Hidden layers
Sigmoid	f(z) = 1/(1 + e^{-z})	Binary classification output
Tanh	f(z) = (e^z - e^{-z})/(e^z + e^{-z})	Hidden layers
Softmax	f(z_i) = e^{z_i}/Σ e^{z_j}	Multi-class output
#### 6️⃣ Loss Functions
Task	Loss Function
Binary Classification	Binary Cross-Entropy
Multi-class Classification	Categorical Cross-Entropy
Regression	Mean Squared Error (MSE)
Binary Cross-Entropy Formula:

math
L = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]
#### 7️⃣ Training MLP → Backpropagation
Training Loop:
text
1. Forward pass → compute Z and A
2. Compute loss
3. Backward pass → compute gradients
4. Update weights (Gradient Descent)
5. Repeat
Gradient Descent Update Rule:
math
W^{[l]} = W^{[l]} - \alpha \frac{\partial L}{\partial W^{[l]}}
math
b^{[l]} = b^{[l]} - \alpha \frac{\partial L}{\partial b^{[l]}}
Where α is the learning rate.

#### 8️⃣ Simple Intuition
Each neuron = simple calculator (weighted sum + activation)

Hidden layers = feature transformers that learn representations

MLP learns complex non-linear relationships through composition

Forward propagation = making predictions

Backpropagation = learning from errors by adjusting weights

#### 📊 Example Code Snippet
python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, weights, biases):
    """
    Simple forward pass for 2-layer MLP
    """
    # Layer 1
    Z1 = np.dot(weights[0], X) + biases[0]
    A1 = sigmoid(Z1)
    
    # Layer 2 (Output)
    Z2 = np.dot(weights[1], A1) + biases[1]
    A2 = sigmoid(Z2)
    
    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
🎯 Key Takeaways
MLPs are universal approximators - can learn any continuous function

Non-linearity from activation functions enables complex pattern learning

Backpropagation efficiently computes gradients using chain rule

Proper weight initialization and regularization are crucial

MLPs form the foundation for more complex architectures (CNNs, RNNs)


