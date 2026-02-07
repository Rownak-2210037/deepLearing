# Multilayer Perceptron(MLP)
### 1.What is an MLP?
MLP = Multi-Layer PerceptronIt is a type of Artificial Neural Network (ANN)
"Multi-layer" → has one or more hidden layers between input and output
Each layer contains neurons (also called nodes or units)
Neurons perform two operations:

Take weighted sum of inputs + bias → Z
Apply activation function → A
### 2.Structure of an MLP
Input layer → Hidden layer(s) → Output layer
Layer Breakdown:
Input layer: Your features/input data

Hidden layer(s): Neurons that learn patterns and representations

Output layer: Final prediction/classification

### 3.Example Architecture:
Input: 4 features → n₀ = 4
Hidden layer: 3 neurons → n₁ = 3
Output: 2 classes → n₂ = 2
### 4.Forward Propagation 
Forward Propagation in MLP 
What is Forward Propagation?
It's the process where input data flows through the neural network layer by layer to produce an output.
#### The 3-Step Process:
##### 1. Input Layer → Hidden Layer
Hidden Neuron Value = Activation( (Input1 × Weight1) + (Input2 × Weight2) + Bias )
Each connection has a weight (strength)
Add a bias (adjustment term)
Apply activation function (like ReLU, Sigmoid) to add non-linearity

##### 2. Hidden Layer → Output Layer
Output = Activation( (Hidden1 × Weight1) + (Hidden2 × Weight2) + Bias )
Same process repeats

Different weights and biases

##### 3. Get Final Output
For regression: Often linear activation

For classification: Softmax for probabilities

### Visual Flow:

Input → Multiply by Weights → Add Bias → Apply Activation → Repeat → Final Output
          (Layer 1)                         (Layer 2)             (Output)

Weights & Biases: Learned during training
Activation Functions: Make network non-linear (can learn complex patterns)
Forward Pass: Just calculation, no learning yet
Learning happens during backpropagation
### Simple Code Snippet:
One layer forward propagation
def forward_layer(inputs, weights, bias, activation):
    z = np.dot(inputs, weights) + bias  # Linear transformation
    return activation(z)  # Non-linear activation
### 5️.Activation Functions (Basics)

<img width="1326" height="278" alt="image" src="https://github.com/user-attachments/assets/522023bc-4331-4c9a-b66e-38d4f409169e" />

### 6.Loss Functions

<img width="880" height="363" alt="image" src="https://github.com/user-attachments/assets/5558f8e3-377a-461c-a849-4e4d3a605c16" />

### 7️.Training MLP → Backpropagation
Training Loop:
1. Forward pass → compute Z and A
2. Compute loss
3. Backward pass → compute gradients
4. Update weights (Gradient Descent)
5. Repeat
Gradient Descent Update Rule:
Gradient Descent updates parameters to minimize the loss function.Move parameters in the opposite direction of the gradient.
<img width="653" height="300" alt="image" src="https://github.com/user-attachments/assets/5979b260-a975-4a19-8799-405d78d05439" />
Where α is the learning rate.

### 8️.Simple Intuition
Each neuron = simple calculator (weighted sum + activation)

Hidden layers = feature transformers that learn representations

MLP learns complex non-linear relationships through composition

Forward propagation = making predictions

Backpropagation = learning from errors by adjusting weights

### 9.Example Code Snippet
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
MLPs are universal approximators - can learn any continuous function

Non-linearity from activation functions enables complex pattern learning

Backpropagation efficiently computes gradients using chain rule

Proper weight initialization and regularization are crucial

MLPs form the foundation for more complex architectures (CNNs, RNNs)

### 1.Code(Forward Propagation)
```
#Forward Code
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation(X, W1, b1, W2, b2):
    """
    X: Input data (n_features, m_samples)
    W1, b1: Weights/biases for layer 1
    W2, b2: Weights/biases for layer 2
    Returns: A2 (predictions), cache (intermediate values)
    """
    # Layer 1
    Z1 = np.dot(W1, X) + b1   # Linear transformation
    A1 = sigmoid(Z1)          # Activation

    # Layer 2 (Output)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # Cache intermediate values (needed for backprop)
    cache = {
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2
    }

    return A2, cache

# Example usage:
n_features = 3
n_hidden = 4
n_output = 1
m_samples = 5

# Initialize parameters
W1 = np.random.randn(n_hidden, n_features) * 0.01
b1 = np.zeros((n_hidden, 1))
W2 = np.random.randn(n_output, n_hidden) * 0.01
b2 = np.zeros((n_output, 1))

# Sample input
X = np.random.randn(n_features, m_samples)

# Forward pass
predictions, cache = forward_propagation(X, W1, b1, W2,b2)
print(f"Predictions shape: {predictions.shape}")
print(f"Sample prediction: {predictions[0, 1]:.4f}")
```
### Output:
**Predictions shape: (1, 5)**

**Sample prediction: 0.5012**


