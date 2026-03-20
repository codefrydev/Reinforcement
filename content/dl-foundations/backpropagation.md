---
title: "Backpropagation: Teaching Networks by Propagating Errors"
description: "The chain rule applied backwards through a neural network — computing gradients for every weight and verifying them with numerical finite differences."
date: 2026-03-20T00:00:00Z
weight: 7
draft: false
difficulty: 5
tags: ["backpropagation", "chain rule", "gradients", "training", "dl-foundations"]
keywords: ["backpropagation", "chain rule neural network", "gradient computation", "numerical gradient check", "DQN training"]
roadmap_icon: "calculator"
roadmap_color: "indigo"
roadmap_phase_label: "Chapter 7"
---

**Learning objectives**

- State the backpropagation algorithm as repeated application of the chain rule.
- Implement forward and backward passes for a 2-layer MLP with MSE loss in NumPy.
- Verify computed gradients against numerical finite differences.

**Concept and real-world motivation**

**Backpropagation** is the algorithm that computes how the loss changes with respect to every weight in the network. It is just the chain rule of calculus, applied efficiently from output to input. Without backpropagation, we could not train deep networks — computing gradients by finite differences would be prohibitively slow for networks with millions of parameters.

For a 2-layer network with MSE loss \\(L = \frac{1}{n}\|\hat{y} - y\|^2\\), the gradients are:

**Output layer:**
\\[\delta_2 = \frac{2}{n}(\hat{y} - y)\\]
\\[\frac{\partial L}{\partial W_2} = \delta_2 \, h_1^T \quad \text{(outer product)}\\]
\\[\frac{\partial L}{\partial b_2} = \delta_2\\]

**Hidden layer (propagating error back through ReLU):**
\\[\delta_1 = (W_2^T \delta_2) \odot \text{ReLU}'(z_1)\\]
\\[\frac{\partial L}{\partial W_1} = \delta_1 \, x^T \quad \text{(outer product)}\\]
\\[\frac{\partial L}{\partial b_1} = \delta_1\\]

where \\(\text{ReLU}'(z) = \mathbb{1}[z > 0]\\) (1 where the pre-activation was positive, 0 elsewhere).

**This is exactly how DQN learns.** The TD error \\((r + \gamma \max_{a'} Q(s', a') - Q(s, a))^2\\) is the loss, and backprop computes gradients of this loss with respect to every weight \\(\theta\\) in the Q-network. The optimizer (usually Adam) then updates \\(\theta \leftarrow \theta - \alpha \nabla_\theta L\\).

**Illustration:**

{{< mermaid >}}
flowchart RL
    L["Loss L"] -->|"δ₂ = 2(ŷ−y)/n"| W2["∂L/∂W₂\n∂L/∂b₂"]
    W2 -->|"δ₁ = W₂ᵀδ₂ ⊙ ReLU'(z₁)"| W1["∂L/∂W₁\n∂L/∂b₁"]
    W1 -->|"∂L/∂x"| Input["Input x"]
{{< /mermaid >}}

**Exercise:** Implement the full forward and backward pass for a 2-layer MLP with MSE loss. Then run a numerical gradient check to verify your backprop implementation is correct.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n\n# Single training example\nx = np.array([[0.5, -0.3, 0.8]])  # shape (1, 3)\ny = np.array([[1.0, 0.0]])        # shape (1, 2)\n\n# Initialize network (3 → 4 → 2, linear output)\nW1 = np.random.randn(4, 3) * 0.5\nb1 = np.zeros(4)\nW2 = np.random.randn(2, 4) * 0.5\nb2 = np.zeros(2)\n\n# --- Forward pass ---\nz1 = x @ W1.T + b1         # shape (1, 4)\nh1 = np.maximum(0, z1)     # ReLU, shape (1, 4)\nz2 = h1 @ W2.T + b2        # shape (1, 2)\ny_hat = z2                 # linear output (MSE regression)\n\nloss = np.mean((y_hat - y) ** 2)\nprint(f'Forward pass loss: {loss:.4f}')\n\n# --- Backward pass ---\nn = x.shape[0]\n\n# TODO: output layer gradient\ndelta2 = None  # (2/n) * (y_hat - y), shape (1, 2)\ndW2    = None  # delta2.T @ h1, shape (2, 4)\ndb2    = None  # delta2.sum(axis=0), shape (2,)\n\n# TODO: hidden layer gradient\nrelu_grad = None  # (z1 > 0).astype(float), shape (1, 4)\ndelta1    = None  # (delta2 @ W2) * relu_grad, shape (1, 4)\ndW1       = None  # delta1.T @ x, shape (4, 3)\ndb1       = None  # delta1.sum(axis=0), shape (4,)\n\nprint(f'dW2 shape: {dW2.shape if dW2 is not None else None}')  # expected: (2, 4)\nprint(f'dW1 shape: {dW1.shape if dW1 is not None else None}')  # expected: (4, 3)\nprint(f'dW1[0,:]: {dW1[0] if dW1 is not None else None}')" height="340" >}}

**Professor's hints**

- `delta2 = (2/n) * (y_hat - y)` — the factor of 2 comes from differentiating \\((\hat{y}-y)^2\\).
- `dW2 = delta2.T @ h1` — the outer product: delta2 is (1,2), h1 is (1,4), so delta2.T @ h1 gives (2,4). ✓
- ReLU gradient: `relu_grad = (z1 > 0).astype(float)` — this is 1 where z1 was positive during the forward pass.
- `delta1 = (delta2 @ W2) * relu_grad` — propagate the error back through W2, then mask with the ReLU gradient.
- `dW1 = delta1.T @ x` — delta1 is (1,4), x is (1,3), so delta1.T @ x gives (4,3). ✓

**Common pitfalls**

- **Wrong ReLU gradient condition:** Using `z1 >= 0` instead of `z1 > 0`. Mathematically, at z=0 the ReLU derivative is undefined (we use the subgradient 0 at exactly zero — this is fine in practice because floating point rarely hits exactly 0).
- **Forgetting to average over the batch:** If processing n examples at once, divide by n. For n=1 this doesn't matter, but for mini-batches it does.
- **Shape errors in outer products:** `delta.T @ h` gives the weight gradient. Make sure the dimensions work out to `(n_out, n_in)` matching the weight matrix shape.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np
np.random.seed(42)

x = np.array([[0.5, -0.3, 0.8]])
y = np.array([[1.0, 0.0]])
W1 = np.random.randn(4, 3) * 0.5
b1 = np.zeros(4)
W2 = np.random.randn(2, 4) * 0.5
b2 = np.zeros(2)

# Forward
z1    = x @ W1.T + b1
h1    = np.maximum(0, z1)
z2    = h1 @ W2.T + b2
y_hat = z2
loss  = np.mean((y_hat - y) ** 2)

# Backward
n       = x.shape[0]
delta2  = (2/n) * (y_hat - y)      # (1, 2)
dW2     = delta2.T @ h1             # (2, 4)
db2     = delta2.sum(axis=0)        # (2,)
delta1  = (delta2 @ W2) * (z1 > 0).astype(float)  # (1, 4)
dW1     = delta1.T @ x             # (4, 3)
db1     = delta1.sum(axis=0)        # (4,)

print(f'loss={loss:.4f}, dW2 shape={dW2.shape}, dW1 shape={dW1.shape}')
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Implement a numerical gradient check for W1[0,0]. Perturb the weight by ±ε, compute loss for each, and estimate the gradient as (L+ − L−) / (2ε). Compare to your backprop gradient.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n\nx = np.array([[0.5, -0.3, 0.8]])\ny = np.array([[1.0, 0.0]])\nW1 = np.random.randn(4, 3) * 0.5\nb1 = np.zeros(4)\nW2 = np.random.randn(2, 4) * 0.5\nb2 = np.zeros(2)\n\ndef forward_loss(W1, b1, W2, b2, x, y):\n    z1 = x @ W1.T + b1\n    h1 = np.maximum(0, z1)\n    z2 = h1 @ W2.T + b2\n    return np.mean((z2 - y) ** 2)\n\n# Numerical gradient for W1[0, 0]\neps = 1e-5\nW1_plus  = W1.copy(); W1_plus[0, 0]  += eps\nW1_minus = W1.copy(); W1_minus[0, 0] -= eps\n\nL_plus  = forward_loss(W1_plus,  b1, W2, b2, x, y)\nL_minus = forward_loss(W1_minus, b1, W2, b2, x, y)\n\nnumerical_grad = (L_plus - L_minus) / (2 * eps)\n\n# Run backprop to get analytical gradient\nz1    = x @ W1.T + b1\nh1    = np.maximum(0, z1)\nz2    = h1 @ W2.T + b2\ny_hat = z2\ndelta2 = (2/1) * (y_hat - y)\ndW2    = delta2.T @ h1\ndelta1 = (delta2 @ W2) * (z1 > 0).astype(float)\ndW1    = delta1.T @ x\n\nanalytical_grad = dW1[0, 0]\n\nprint(f'Numerical gradient:  {numerical_grad:.6f}')\nprint(f'Analytical gradient: {analytical_grad:.6f}')\nprint(f'Relative error: {abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + 1e-8):.2e}')\n# expected: relative error < 1e-5" height="300" >}}

2. **Coding:** Implement a full training loop: run forward pass, backward pass, and gradient descent update (`W -= lr * dW`) for 100 iterations. Print the loss every 20 steps and verify it decreases.
3. **Challenge:** Extend the backprop implementation to a 3-layer network (add a second hidden layer). Write out the gradient formulas for the new layer before implementing.
4. **Variant:** Change the loss from MSE to mean absolute error (MAE): \\(L = \frac{1}{n}\sum|y - \hat{y}|\\). How does the gradient \\(\delta_2\\) change? Implement and compare convergence speed.
5. **Debug:** The backprop below uses the wrong condition for the ReLU gradient — it uses `>= 0` instead of the subgradient convention `> 0`. While both work in practice, the comment in the code incorrectly claims `>= 0` is always correct. More importantly, there's a sign error in `delta2`. Find and fix both issues.

{{< pyrepl code="import numpy as np\nnp.random.seed(0)\n\nx = np.array([[1.0, 0.5]])\ny = np.array([[0.0]])\nW1 = np.array([[0.5, -0.3], [0.2, 0.8]])\nb1 = np.zeros(2)\nW2 = np.array([[0.4, -0.6]])\nb2 = np.zeros(1)\n\nz1    = x @ W1.T + b1\nh1    = np.maximum(0, z1)\nz2    = h1 @ W2.T + b2\ny_hat = z2\nloss  = np.mean((y_hat - y)**2)\n\n# BUG: delta2 sign is wrong — should be (y_hat - y), not (y - y_hat)\ndelta2 = (2/1) * (y - y_hat)   # TODO: fix sign\n# BUG: relu gradient should use > 0, not >= 0 (minor, but fix the comment)\nrelu_grad = (z1 >= 0).astype(float)  # TODO: change to z1 > 0\n\ndW2    = delta2.T @ h1\ndb2    = delta2.sum(axis=0)\ndelta1 = (delta2 @ W2) * relu_grad\ndW1    = delta1.T @ x\n\n# After fix, verify that loss decreases after one gradient step\nlr = 0.1\nW1_new = W1 - lr * dW1\nW2_new = W2 - lr * dW2\n\nnew_z2   = np.maximum(0, x @ W1_new.T + b1) @ W2_new.T + b2\nnew_loss = np.mean((new_z2 - y)**2)\nprint(f'Old loss: {loss:.4f}')\nprint(f'New loss: {new_loss:.4f}')\n# expected: new_loss < loss (gradient descent decreases loss)" height="280" >}}

6. **Conceptual:** Why can't we use the gradient 0 everywhere for the ReLU derivative (setting it to 0 at z=0 and everywhere else too)? This is the "dead neuron" argument — explain what happens to the weights when the ReLU gradient is always 0. The term "subgradient" refers to the convention we use at z=0.
7. **Recall:** Write the four backpropagation equations for a 2-layer MSE network from memory: \\(\delta_2\\), \\(\partial L/\partial W_2\\), \\(\delta_1\\), and \\(\partial L/\partial W_1\\). What does each equation compute, and why does the order (output → input) matter?
