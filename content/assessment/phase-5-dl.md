---
title: "Phase 5 Assessment: Deep Learning Foundations"
description: "12 questions covering neural networks, backpropagation, training loops, and CNNs. Pass: 9/12."
date: 2026-03-20T00:00:00Z
draft: false
tags: ["assessment", "phase 5", "deep learning", "neural networks", "dl-foundations", "self-check"]
keywords: ["phase 5 assessment", "deep learning quiz", "neural network check", "backprop quiz"]
weight: 7
roadmap_icon: "sparkles"
roadmap_color: "purple"
roadmap_phase_label: "Phase 5 Quiz"
---

Use this self-check after completing [DL Foundations](../dl-foundations/). Pass: **9 out of 12**. If you score below 9, review the topics you missed before continuing to Phase 6 (RL Foundations).

---

### 1. Predict the output

Forward pass through a 2-layer network with \\(x=[1, -1]\\):
- \\(W_1 = \begin{bmatrix}2&0\\0&2\end{bmatrix}\\), \\(b_1 = [0, 0]\\), activation = ReLU
- \\(W_2 = [1, 1]\\), \\(b_2 = 0\\), no activation (linear output)

What is the final output?

{{< collapse summary="Answer" >}}
**Layer 1:** \\(z_1 = W_1 x + b_1 = [2\cdot1+0\cdot(-1),\ 0\cdot1+2\cdot(-1)] = [2, -2]\\)

ReLU: \\(a_1 = [2, 0]\\) (−2 is zeroed out)

**Layer 2:** \\(z_2 = W_2 a_1 + b_2 = 1\cdot2 + 1\cdot0 + 0 = \mathbf{2}\\)

Key insight: ReLU killed the second neuron — its information is lost. This is the "dead neuron" behavior.
{{< /collapse >}}

---

### 2. Write a function

Implement ReLU and its backward pass.

{{< pyrepl code="import numpy as np\n\ndef relu(z):\n    return np.maximum(0, z)\n\ndef relu_backward(dz, z):\n    # dz: gradient from upstream, z: pre-activation values\n    return dz * (z > 0)\n\n# Test\nz = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])\ndz = np.ones_like(z)  # gradient of 1 from upstream\nprint('relu(z):         ', relu(z))\nprint('relu_backward:   ', relu_backward(dz, z))  # expected: [0,0,0,1,1]" height="200" >}}

{{< collapse summary="Answer" >}}
```python
def relu(z):
    return np.maximum(0, z)

def relu_backward(dz, z):
    return dz * (z > 0)
```
ReLU backward: gradient passes through where pre-activation z>0; blocked (×0) where z≤0. The `z>0` comparison gives a binary mask (True/False = 1/0).
{{< /collapse >}}

---

### 3. Find the bug

The softmax below is numerically unstable. Find and fix it.

```python
import numpy as np

def softmax(z):
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
```

{{< collapse summary="Answer" >}}
Bug: when `z` contains large values (e.g. z=1000), `np.exp(1000)` overflows to `inf`.

Fix: subtract the row maximum before exponentiating:
```python
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
```
This is mathematically equivalent (multiplying numerator and denominator by the same constant \\(e^{-\max}\\)), but avoids overflow. The result is the same probability distribution.
{{< /collapse >}}

---

### 4. Predict the output

After applying sigmoid activation, what are the constraints on the output?

{{< collapse summary="Answer" >}}
Sigmoid output is always in the open interval **(0, 1)**:
- Always positive (never 0 or 1 exactly, only approaching them)
- \\(\sigma(z) > 0.5\\) for z > 0; \\(\sigma(z) < 0.5\\) for z < 0; \\(\sigma(0) = 0.5\\)
- Monotonically increasing: larger z → larger output

For a vector of outputs (multi-class), sigmoid applied element-wise does **not** make outputs sum to 1 — use softmax instead if you need a probability distribution.
{{< /collapse >}}

---

### 5. Write a function

Implement MSE loss.

{{< pyrepl code="import numpy as np\n\ndef mse_loss(y_pred, y_true):\n    return np.mean((y_pred - y_true) ** 2)\n\n# Test\npred = np.array([0.8, 0.2, 0.6])\ntrue = np.array([1.0, 0.0, 1.0])\nprint('MSE:', mse_loss(pred, true))  # expected: (0.04+0.04+0.16)/3 ≈ 0.0800" height="160" >}}

{{< collapse summary="Answer" >}}
```python
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)
```
MSE = mean of squared errors. Its gradient with respect to predictions is `2 * (y_pred - y_true) / n`. In DQN, MSE between predicted Q-values and TD targets is the loss that drives Q-network training.
{{< /collapse >}}

---

### 6. Find the bug

This backpropagation code computes \\(\delta_1\\) (gradient for the first hidden layer) incorrectly.

```python
# Forward pass:
z1 = X @ W1.T + b1
a1 = relu(z1)
z2 = a1 @ W2.T + b2
a2 = softmax(z2)

# Backward pass:
delta2 = (a2 - Y) / n
dW2 = delta2.T @ a1
delta1 = (delta2 @ W1) * (z1 > 0)   # bug
dW1 = delta1.T @ X
```

{{< collapse summary="Answer" >}}
Bug: `delta2 @ W1` should be `delta2 @ W2`.

To propagate gradients from layer 2 back to layer 1, you multiply by the **transposed weight matrix of layer 2** (W2), not layer 1 (W1). Using W1 would produce wrong gradients (wrong dimensions and wrong values).

Fix: `delta1 = (delta2 @ W2) * (z1 > 0)`

The chain rule: \\(\delta_1 = (\delta_2 W_2) \odot \text{ReLU}'(z_1)\\).
{{< /collapse >}}

---

### 7. Conceptual

Explain the chain rule in the context of backpropagation. Why is it the key mathematical insight?

{{< collapse summary="Answer" >}}
The **chain rule** says: if \\(L = f(g(x))\\), then \\(\frac{dL}{dx} = \frac{dL}{dg} \cdot \frac{dg}{dx}\\).

In a neural network, each layer applies a function to the previous layer's output. The loss is a composition of all these functions. To find \\(\frac{\partial L}{\partial w_1}\\) (gradient w.r.t. first layer weights), we must "chain" through all subsequent layers:

\\(\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdots \frac{\partial a_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial w_1}\\)

Backpropagation efficiently computes this by passing "error signals" (deltas) from the output layer backward, reusing computations instead of re-deriving gradients for each layer from scratch.
{{< /collapse >}}

---

### 8. Predict the output

What happens when the learning rate is 10.0? What happens when it is 0.0001?

{{< collapse summary="Answer" >}}
**Learning rate = 10.0 (too large):**
- Weights overshoot the minimum — the update jumps past the optimal point
- Loss oscillates or diverges (increases instead of decreasing)
- Training becomes unstable or fails entirely

**Learning rate = 0.0001 (too small):**
- Weights update very slowly — convergence is painfully slow
- May appear to not be learning at all over a normal number of epochs
- Can get stuck in local minima near initialization

The right learning rate balances speed and stability. Common starting points: 1e-3 for Adam, 1e-2 for SGD. Always check the loss curve — should decrease smoothly.
{{< /collapse >}}

---

### 9. Write a function

Implement `one_hot(y, n_classes)`.

{{< pyrepl code="import numpy as np\n\ndef one_hot(y, n_classes):\n    Y = np.zeros((len(y), n_classes))\n    Y[np.arange(len(y)), y] = 1\n    return Y\n\n# Test\ny = np.array([0, 2, 1])\nresult = one_hot(y, n_classes=3)\nprint(result)\n# expected:\n# [[1,0,0],\n#  [0,0,1],\n#  [0,1,0]]" height="180" >}}

{{< collapse summary="Answer" >}}
```python
def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1
    return Y
```
Advanced indexing: `Y[row_indices, col_indices] = 1` sets exactly the right position in each row to 1. Used for computing cross-entropy loss and for the output layer gradient in classification.
{{< /collapse >}}

---

### 10. Find the bug

Dropout is being applied during test time, causing non-deterministic evaluation.

```python
import numpy as np

def forward(X, W, b, training=True, p_drop=0.5):
    z = X @ W + b
    a = np.maximum(0, z)  # ReLU
    mask = (np.random.rand(*a.shape) > p_drop)   # bug
    a = a * mask / (1 - p_drop)
    return a
```

{{< collapse summary="Answer" >}}
Bug: the dropout mask is applied regardless of the `training` flag. At test time (`training=False`), dropout should be disabled — all neurons should be active.

Fix:
```python
def forward(X, W, b, training=True, p_drop=0.5):
    z = X @ W + b
    a = np.maximum(0, z)
    if training:   # only apply dropout during training
        mask = (np.random.rand(*a.shape) > p_drop)
        a = a * mask / (1 - p_drop)
    return a
```
Without this fix, evaluation results are random — different calls give different predictions for the same input.
{{< /collapse >}}

---

### 11. Conceptual

Why does DQN use a target network? Connect to training stability.

{{< collapse summary="Answer" >}}
In DQN, the TD target is \\(y = r + \gamma \max_a Q(s', a; \theta)\\). If we use the **same** network for both predictions (Q(s,a)) and targets (max Q(s')), both sides of the loss change with every update — the target is a moving goalposts problem.

The **target network** is a frozen copy of the Q-network with parameters \\(\theta^-\\) updated only every \\(C\\) steps (e.g. every 10,000 steps). During training, targets use \\(\theta^-\\): \\(y = r + \gamma \max_a Q(s', a; \theta^-)\\). This makes targets stable for \\(C\\) steps, drastically reducing oscillation.

**Connection to DL:** In supervised learning, targets are fixed (human labels). DQN's target network simulates fixed labels for short windows, making the RL training loop resemble supervised learning within those windows.
{{< /collapse >}}

---

### 12. Predict the output

For L2 regularization with penalty \\(L_{reg} = L + \frac{\lambda}{2}\|w\|^2\\), how does the gradient change?

{{< collapse summary="Answer" >}}
The gradient of the regularized loss is:

\\(\nabla L_{reg} = \nabla L + \lambda w\\)

The L2 term adds \\(\lambda w\\) to the gradient. In the SGD update:

\\(w \leftarrow w - \alpha(\nabla L + \lambda w) = w(1 - \alpha\lambda) - \alpha\nabla L\\)

The factor \\((1 - \alpha\lambda)\\) **shrinks** the weights slightly at every step (weight decay). This prevents weights from growing very large and forces the model to use smaller, more distributed representations. For \\(\alpha=0.01, \lambda=0.01\\): each step multiplies the weight by \\(0.9999\\) before subtracting the gradient.
{{< /collapse >}}

---

**Score:** 9–12: Ready for Phase 6 (RL Foundations — Volume 1). 7–8: Review the specific DL Foundations pages covering your missed topics. Below 7: Complete [DL Foundations](../dl-foundations/) before continuing.
