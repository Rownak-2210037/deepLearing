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

### 1.1.Code(Forward Propagation using numpy)
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

### 1.2.Code(Forward with pytorch)
```
#forward with pytorch
import torch
import torch.nn as nn

# Sample input
X = torch.randn(5, 3)       # 5 samples, 3 features
Y = torch.tensor([[1,0,1,0,1]]).float().T  # shape (5,1)

# Define MLP
model = nn.Sequential(
    nn.Linear(3, 4),   # input 3 features → hidden 4 neurons
    nn.Sigmoid(),
    nn.Linear(4, 1),   # hidden 4 neurons → output 1 neuron
    nn.Sigmoid()
)

# Forward pass
predictions = model(X)  # shape (5,1)
print("Predictions:\n", predictions)
print("Predictions shape:", predictions.shape)

```
### Output:
**Predictions:**
 **tensor([[0.5491],
        [0.5146],
        [0.5188],
        [0.5401],
        [0.5292]], grad_fn=<SigmoidBackward0>)**
        
**Predictions shape: torch.Size([5, 1])**

#### --->Note:
**About “Z1 = W1·X + b1” vs “X·W1 + b1”**

**NumPy convention: usually Z1 = W1·X + b1**

**W1 shape = (hidden, input)**

**X shape = (input, samples)**

**PyTorch convention: Z1 = X·W1^T + b1 automatically handled by nn.Linear**

**X shape = (samples, features)**

**nn.Linear(features_in, features_out) → PyTorch transposes weights internally**



## 10.Backpropagation in Neural Networks 
#### 10.1 What is Backpropagation?

Backpropagation is the process by which a neural network learns from its mistakes.
Forward propagation → makes a prediction
Backpropagation → measures the error and updates weights & biases
The goal is to minimize the loss function.

#### 10.2. Why Do We Need Backpropagation?

A neural network has many weights.
Backpropagation answers two key questions:

**Which weight caused how much error?**

**Should a weight be increased or decreased?**

It uses gradients to update parameters in the correct direction.

#### 10.3. High-Level Idea

Do a forward pass and get prediction
Compute the loss
Propagate the error backwards through the network
Update weights using gradient descent
This backward flow of error gives the name Backpropagation.

#### 10.4. Forward Propagation (Recap)

For a simple MLP with one hidden layer:

Z1 = W1 · X + b1
A1 = σ(Z1)

Z2 = W2 · A1 + b2
A2 = σ(Z2)
Where:

Z = linear output

A = activation output

σ = activation function (Sigmoid)

#### 10.5. Loss Function (Binary Classification)
Loss = -[ y log(A2) + (1 - y) log(1 - A2) ]

This measures how wrong the prediction is.

#### 10.6. Backpropagation Steps
**Step 1: Error at Output Layer**
dZ2 = A2 - Y
This is the difference between prediction and actual label.

**Step 2: Gradients for Output Layer**
dW2 = (1/m) · dZ2 · A1ᵀ
db2 = (1/m) · sum(dZ2)

**Step 3: Error at Hidden Layer**
dZ1 = W2ᵀ · dZ2 ⊙ σ′(Z1)

The output error is propagated backward using the chain rule.

**Step 4: Gradients for Hidden Layer**
dW1 = (1/m) · dZ1 · Xᵀ
db1 = (1/m) · sum(dZ1)

#### 10.7. Weight Update Rule (Gradient Descent)
W = W - α · dW
b = b - α · db
Where:

α = learning rate

This step allows the network to learn.

#### 10.8. Why Do We Store Values in Cache?

During forward propagation, we compute:

Z1, A1, Z2, A2
Backpropagation needs these values again to compute gradients.

So we store them in a cache:

cache = {
    "Z1": Z1, "A1": A1,
    "Z2": Z2, "A2": A2
}
This avoids recomputation and keeps backprop clean and efficient.

#### 10.9. One-Line Summary

Backpropagation uses the chain rule to compute gradients of the loss with respect to network parameters and updates them to reduce prediction error.

#### 10.10. Final Intuition

Forward pass → prediction

Backward pass → correction

Repeated many times → learning

This is how an MLP learns from data.



### 2.1.Code(Backward Propagation using numpy)
```
#Backward with numpy
import numpy as np

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (for backprop)
def sigmoid_derivative(a):
    return a * (1 - a)

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache={
          "Z1":Z1,
          "A1":A1,
          "Z2":Z2,
          "A2":A2
          }
    
    return A2, cache

# Backward propagation
def backward_propagation(X, Y, cache, W2):
    m = X.shape[1]  # number of samples

    A1 = cache["A1"]
    A2 = cache["A2"]

    # Output layer error
    dZ2 = A2 - Y           # For binary cross-entropy with sigmoid
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer error
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(A1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1,
             "dW2": dW2, "db2": db2}

    return grads

# Update parameters
def update_parameters(W1, b1, W2, b2, grads, lr=0.1):
    W1 -= lr * grads["dW1"]
    b1 -= lr * grads["db1"]
    W2 -= lr * grads["dW2"]
    b2 -= lr * grads["db2"]
    return W1, b1, W2, b2

# Example usage
n_features = 3
n_hidden = 4
n_output = 1
m_samples = 5

# Initialize
np.random.seed(1)
W1 = np.random.randn(n_hidden, n_features) * 0.01
b1 = np.zeros((n_hidden, 1))
W2 = np.random.randn(n_output, n_hidden) * 0.01
b2 = np.zeros((n_output, 1))

X = np.random.randn(n_features, m_samples)
Y = np.array([[1, 0, 1, 0, 1]])  # sample labels

# Forward pass
A2, cache = forward_propagation(X, W1, b1, W2, b2)

# Backward pass
grads = backward_propagation(X, Y, cache, W2)

# Update parameters
W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, grads, lr=0.1)

print("Updated W1:\n", W1)
print("Updated W2:\n", W2)
print("Sample predictions:\n", A2)
```
### Output:

![Screenshot 2026-02-10 020951](https://github.com/user-attachments/assets/e2f51c11-cf4a-4433-8422-bc022d091c63)


### 2.2.Code(Backward with pytorch)
```
import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(5, 3)  # 5 samples, 3 features
Y = torch.tensor([[1,0,1,0,1]]).float().T  # shape (5,1)

#calculate z1,A1,Z2,A2...
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
 
criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop

for i in range(10):

    # Forward pass
    predictions = model(X)

    # Compute loss
    loss = criterion(predictions, Y)

    # Backward pass
    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # compute gradients
    optimizer.step()        # update weights

    print(f"Iteration: {i}, Loss: {loss.item()}")

print("\nFinal Predictions:\n", predictions)

```
### Output:
![Screenshot 2026-02-10 020951](https://github.com/user-attachments/assets/d268aa71-8917-4371-9a7c-d5ac3c457244)




