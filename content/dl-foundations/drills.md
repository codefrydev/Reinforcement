---
title: "DL Foundations Drills"
description: "15 drill problems covering neural networks, forward pass, backpropagation, optimizers, and training."
date: 2026-03-20T00:00:00Z
weight: 99
draft: false
difficulty: 5
tags: ["drills", "deep learning", "neural networks", "backpropagation", "dl-foundations"]
keywords: ["DL drills", "neural network exercises", "backprop practice", "optimizer drill", "forward pass drill"]
roadmap_icon: "book"
roadmap_color: "rose"
roadmap_phase_label: "Drills"
---

Quick-fire practice for DL Foundations. Work through these after completing the main pages. Answers in the collapses; pyrepl blocks for coding problems.

---

## Recall (R)

**R1.** What is the vanishing gradient problem? When does it occur?

{{< collapse summary="Answer" >}}
The vanishing gradient problem occurs when gradients become extremely small as they propagate backward through many layers. Each layer multiplies the gradient by the derivative of its activation function — for sigmoid/tanh, derivatives are at most 0.25/1, so stacking many layers shrinks the gradient toward zero. Earlier layers receive nearly no learning signal.

It occurs most severely with sigmoid or tanh activations in deep networks (many layers). ReLU largely solves it because ReLU's derivative is 1 for positive inputs (no squashing).
{{< /collapse >}}

---

**R2.** Write the ReLU function and its derivative.

{{< collapse summary="Answer" >}}
\\(\text{ReLU}(z) = \max(0, z)\\)

Derivative: \\(\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}\\)

In NumPy: `relu = lambda z: np.maximum(0, z)` and `relu_backward = lambda dz, z: dz * (z > 0)`.
{{< /collapse >}}

---

**R3.** What is the purpose of bias terms in a neural network?

{{< collapse summary="Answer" >}}
Bias terms allow the network to shift the activation function horizontally — they allow the neuron to fire (output non-zero) even when all inputs are zero. Without biases, every hyperplane the network learns would pass through the origin, severely limiting expressivity. Biases give each neuron an independent threshold.
{{< /collapse >}}

---

**R4.** What is the difference between a batch and an epoch?

{{< collapse summary="Answer" >}}
- **Batch (mini-batch):** A subset of the training data processed in one forward-backward pass. Typical sizes: 32, 64, 128 samples.
- **Epoch:** One complete pass through the entire training dataset. Multiple epochs = the data is seen many times.

One epoch = (dataset size / batch size) gradient update steps. If dataset has 1000 samples and batch size is 100, one epoch = 10 gradient steps.
{{< /collapse >}}

---

**R5.** Why do we need activation functions? What happens without them?

{{< collapse summary="Answer" >}}
Activation functions introduce **non-linearity**. Without them, a stack of linear layers collapses to a single linear transformation: \\(W_2(W_1 x + b_1) + b_2 = W' x + b'\\). No matter how many layers you add, you can only represent linear functions — XOR and most real-world problems are non-linear.

Activation functions (ReLU, sigmoid, tanh) allow the network to learn non-linear decision boundaries.
{{< /collapse >}}

---

## Compute (C)

**C1.** Forward pass: \\(x=[1,0]\\), \\(W=\begin{bmatrix}1&2\\3&4\end{bmatrix}\\), \\(b=[0,0]\\), activation = ReLU. Compute \\(h = \text{ReLU}(Wx+b)\\).

{{< collapse summary="Answer" >}}
\\(Wx + b = \begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} + \begin{bmatrix}0\\0\end{bmatrix} = \begin{bmatrix}1\\3\end{bmatrix}\\)

\\(h = \text{ReLU}\begin{bmatrix}1\\3\end{bmatrix} = \begin{bmatrix}1\\3\end{bmatrix}\\) (both positive, unchanged)
{{< /collapse >}}

---

**C2.** MSE loss for predictions \\([0.8, 0.2]\\) and true values \\([1, 0]\\).

{{< collapse summary="Answer" >}}
\\(\text{MSE} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2 = \frac{(1-0.8)^2 + (0-0.2)^2}{2} = \frac{0.04 + 0.04}{2} = 0.04\\)
{{< /collapse >}}

---

**C3.** Softmax for logits \\([2, 1, 0.5]\\). Show each step.

{{< collapse summary="Answer" >}}
Step 1 — exponentiate: \\(e^2 \approx 7.389\\), \\(e^1 \approx 2.718\\), \\(e^{0.5} \approx 1.649\\).

Step 2 — sum: \\(7.389 + 2.718 + 1.649 \approx 11.756\\).

Step 3 — divide: \\([7.389/11.756, 2.718/11.756, 1.649/11.756] \approx [0.629, 0.231, 0.140]\\).

Check: \\(0.629 + 0.231 + 0.140 = 1.000\\) ✓
{{< /collapse >}}

---

**C4.** SGD update: \\(w=1.5\\), gradient=\\(0.3\\), \\(lr=0.1\\). What is \\(w_{new}\\)?

{{< collapse summary="Answer" >}}
\\(w_{new} = w - lr \cdot \nabla L = 1.5 - 0.1 \times 0.3 = 1.5 - 0.03 = \mathbf{1.47}\\)
{{< /collapse >}}

---

**C5.** L2 penalty for \\(w=[2, -1, 0.5]\\) with \\(\lambda=0.01\\).

{{< collapse summary="Answer" >}}
\\(L_2 = \frac{\lambda}{2}\|w\|^2 = \frac{0.01}{2}(4 + 1 + 0.25) = 0.005 \times 5.25 = \mathbf{0.02625}\\)
{{< /collapse >}}

---

## Code (K)

**K1.** Implement `relu_backward(dz, z)` — gradient of ReLU.

{{< pyrepl code="import numpy as np\n\ndef relu_backward(dz, z):\n    # dz: gradient flowing back from next layer\n    # z: pre-activation values (before ReLU)\n    # Return: gradient after passing through ReLU\n    return dz * (z > 0)\n\n# Test\nz = np.array([-1.0, 0.5, 2.0, -0.3])\ndz = np.array([1.0, 1.0, 1.0, 1.0])\nresult = relu_backward(dz, z)\nprint('relu_backward:', result)  # expected: [0, 1, 1, 0]" height="200" >}}

{{< collapse summary="Answer" >}}
```python
def relu_backward(dz, z):
    return dz * (z > 0)
```
The gradient of ReLU is 1 where the input was positive, 0 where it was negative (or zero). Multiply element-wise by the incoming gradient `dz` (chain rule).
{{< /collapse >}}

---

**K2.** Implement `one_hot(y, n_classes)` that converts integer labels to one-hot vectors.

{{< pyrepl code="import numpy as np\n\ndef one_hot(y, n_classes):\n    Y = np.zeros((len(y), n_classes))\n    Y[np.arange(len(y)), y] = 1\n    return Y\n\n# Test\ny = np.array([0, 2, 1, 3])\nresult = one_hot(y, n_classes=4)\nprint(result)\n# expected:\n# [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]" height="180" >}}

{{< collapse summary="Answer" >}}
```python
def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1
    return Y
```
`np.arange(len(y))` selects each row, `y` indexes the column — this places a 1 in the correct position for each sample simultaneously.
{{< /collapse >}}

---

## Debug (D)

**D1.** Backprop bug: gradients for W2 computed before gradients for the output layer.

{{< collapse summary="Answer" >}}
In backprop, you must process layers in **reverse order**: output layer first, then hidden layers. Computing `dW2` requires `delta2` (the gradient at the output), and computing `dW1` requires `delta1` which depends on `delta2` and `W2`. If you compute `dW2 = delta1.T @ a1` before computing `delta2`, `delta1` doesn't exist yet.

Fix: always follow the chain rule order — compute output layer gradients first, then propagate backward:
```python
# CORRECT order:
delta2 = (a2 - Y) / n          # 1. output layer gradient
dW2 = delta2.T @ a1             # 2. W2 gradient
delta1 = (delta2 @ W2) * (z1 > 0)  # 3. hidden layer gradient
dW1 = delta1.T @ X              # 4. W1 gradient
```
{{< /collapse >}}

---

**D2.** Adam bug: bias correction terms missing (\\(\hat{m}\\) and \\(\hat{v}\\) not computed).

{{< collapse summary="Answer" >}}
Without bias correction, in the first few steps Adam's update is far too small. At step \\(t=1\\) with \\(\beta_1=0.9\\): \\(m = 0.1 \cdot g\\). The update uses \\(m\\) directly instead of \\(\hat{m} = m / (1-0.9) = m / 0.1 = g\\). So the first step is 10× smaller than intended.

Fix:
```python
# After updating m and v:
m_hat = m / (1 - beta1 ** t)   # bias-corrected first moment
v_hat = v / (1 - beta2 ** t)   # bias-corrected second moment
w -= lr * m_hat / (np.sqrt(v_hat) + eps)
```
{{< /collapse >}}

---

## Challenge (X)

**X1.** Implement a 3-layer MLP with Adam optimizer. Train on XOR data (4 samples) for 1000 epochs. XOR: inputs [(0,0),(0,1),(1,0),(1,1)], outputs [0,1,1,0].

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n\n# XOR data\nX = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)\ny = np.array([0, 1, 1, 0], dtype=float).reshape(-1, 1)\n\n# 3-layer MLP: 2->4->4->1\nW1 = np.random.randn(4, 2) * 0.5\nb1 = np.zeros((1, 4))\nW2 = np.random.randn(4, 4) * 0.5\nb2 = np.zeros((1, 4))\nW3 = np.random.randn(1, 4) * 0.5\nb3 = np.zeros((1, 1))\n\ndef relu(z): return np.maximum(0, z)\ndef sigmoid(z): return 1 / (1 + np.exp(-z))\n\n# Adam params\nlr = 0.01\nbeta1, beta2, eps = 0.9, 0.999, 1e-8\nparams = [W1, b1, W2, b2, W3, b3]\nm_list = [np.zeros_like(p) for p in params]\nv_list = [np.zeros_like(p) for p in params]\n\nfor t in range(1, 1001):\n    # Forward\n    z1 = X @ W1.T + b1; a1 = relu(z1)\n    z2 = a1 @ W2.T + b2; a2 = relu(z2)\n    z3 = a2 @ W3.T + b3; a3 = sigmoid(z3)\n    loss = np.mean((a3 - y) ** 2)\n    # Backprop\n    d3 = 2 * (a3 - y) / len(X) * a3 * (1 - a3)\n    dW3 = d3.T @ a2; db3 = d3.sum(0, keepdims=True)\n    d2 = (d3 @ W3) * (z2 > 0)\n    dW2 = d2.T @ a1; db2 = d2.sum(0, keepdims=True)\n    d1 = (d2 @ W2) * (z1 > 0)\n    dW1 = d1.T @ X; db1 = d1.sum(0, keepdims=True)\n    grads = [dW1, db1, dW2, db2, dW3, db3]\n    # Adam update\n    for i, (p, g) in enumerate(zip(params, grads)):\n        m_list[i] = beta1 * m_list[i] + (1 - beta1) * g\n        v_list[i] = beta2 * v_list[i] + (1 - beta2) * g**2\n        m_hat = m_list[i] / (1 - beta1**t)\n        v_hat = v_list[i] / (1 - beta2**t)\n        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)\n    W1,b1,W2,b2,W3,b3 = params\nprint(f'Final loss: {loss:.6f}')\nprint('Predictions:', a3.flatten().round(3), '  expected ~[0,1,1,0]')" height="340" >}}

{{< collapse summary="Answer" >}}
XOR requires at least one hidden layer — it's not linearly separable. After ~1000 epochs with Adam and 3 layers, the network converges: outputs ≈ [0, 1, 1, 0].

Key insight: if loss doesn't decrease, try larger initial weights (`* 1.0` instead of `* 0.5`) or a higher learning rate.
{{< /collapse >}}
