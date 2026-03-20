---
title: "Forward Propagation: Computing the Network Output"
description: "Layer-by-layer forward pass through an MLP — computing pre-activations, applying activations, and understanding intermediate representations."
date: 2026-03-20T00:00:00Z
weight: 5
draft: false
difficulty: 5
tags: ["forward propagation", "forward pass", "neural network", "layers", "dl-foundations"]
keywords: ["forward propagation", "forward pass neural network", "intermediate activations", "batch forward pass", "DQN inference"]
roadmap_icon: "trend-up"
roadmap_color: "purple"
roadmap_phase_label: "Chapter 5"
---

**Learning objectives**

- Implement the full forward pass of a 2-layer MLP in NumPy, computing and storing every intermediate value.
- Extend the forward pass to a batch of inputs using matrix operations.
- Identify the shapes of all intermediate tensors in a forward pass and explain why storing them is necessary for backpropagation.

**Concept and real-world motivation**

**Forward propagation** is the computation that turns an input into a network output. It is the inference step: given the current network weights, what is the predicted output for this input? Every time you call a trained model to make a prediction, a forward pass runs. Every training step runs forward propagation first, then backpropagation.

For a 2-layer MLP:
\\[z_1 = W_1 x + b_1\\]
\\[h_1 = \text{ReLU}(z_1)\\]
\\[z_2 = W_2 h_1 + b_2\\]
\\[\hat{y} = \text{softmax}(z_2)\\]

Each \\(z_\ell\\) is called the **pre-activation** (or logit). Each \\(h_\ell\\) is the **post-activation** (or hidden representation). The pre-activations must be **stored** during the forward pass because backpropagation needs them to compute gradients — specifically, the ReLU derivative \\(\text{ReLU}'(z) = \mathbb{1}[z > 0]\\) requires knowing the sign of each pre-activation.

**During inference in DQN**, the forward pass computes Q-values for all actions given the current state:
\\[Q(s, \cdot ; \theta) = W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 s + b_1) + b_2) + b_3\\]
The agent then picks the action with the highest Q-value: \\(a^* = \arg\max_a Q(s, a)\\). This entire computation is a single forward pass.

**Illustration:**

{{< mermaid >}}
flowchart LR
    X["Input x\n(3,)"] --> Z1["z₁ = W₁x + b₁\n(4,)"]
    Z1 --> H1["h₁ = ReLU(z₁)\n(4,)"]
    H1 --> Z2["z₂ = W₂h₁ + b₂\n(2,)"]
    Z2 --> Y["ŷ = softmax(z₂)\n(2,)"]
{{< /mermaid >}}

**Exercise:** Implement the full forward pass for a 2-layer MLP. Print each intermediate value with its shape to see how the representation evolves layer by layer.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n\n# Input: 3 features\nx = np.array([0.5, -0.3, 0.8])\n\n# Layer 1: 3 → 4 neurons (ReLU)\nW1 = np.random.randn(4, 3)\nb1 = np.random.randn(4)\n\n# Layer 2: 4 → 2 neurons (softmax output)\nW2 = np.random.randn(2, 4)\nb2 = np.random.randn(2)\n\n# TODO: compute z1 (pre-activation of layer 1)\nz1 = None  # W1 @ x + b1\n\n# TODO: compute h1 (post-activation, ReLU)\nh1 = None  # np.maximum(0, z1)\n\n# TODO: compute z2 (pre-activation of layer 2)\nz2 = None  # W2 @ h1 + b2\n\n# TODO: compute output with softmax\ndef softmax(z):\n    e = np.exp(z - np.max(z))\n    return e / np.sum(e)\n\ny_hat = None  # softmax(z2)\n\nprint(f'x     shape={x.shape},    values={x}')\nprint(f'z1    shape={z1.shape},   values={np.round(z1,3)}')\nprint(f'h1    shape={h1.shape},   values={np.round(h1,3)}')\nprint(f'z2    shape={z2.shape},   values={np.round(z2,3)}')\nprint(f'y_hat shape={y_hat.shape}, values={np.round(y_hat,4)}')\nprint(f'sum(y_hat)={np.sum(y_hat):.4f}')\n# expected: y_hat sums to 1.0" height="300" >}}

**Professor's hints**

- `W1 @ x` does matrix-vector multiplication (same as `np.dot(W1, x)` for 2D × 1D).
- After layer 1: `z1 = W1 @ x + b1` has shape (4,). After ReLU: `h1` also has shape (4,), but some elements may be zeroed.
- The shapes chain: (3,) → (4,) → (4,) → (2,) → (2,). Each arrow is one layer's operation.
- Store all intermediate values — you'll need them in backpropagation.

**Common pitfalls**

- **Broadcasting error with bias:** If `b1` has shape `(4, 1)` instead of `(4,)`, adding it to `W1 @ x` (shape `(4,)`) will broadcast incorrectly. Keep biases as 1D arrays.
- **Not storing intermediates:** In training mode, you need `z1` and `z2` for backprop. A clean implementation stores them in a dict or as named variables.
- **Applying softmax to hidden layers:** Softmax goes on the output layer only (for classification). Using it in hidden layers prevents neurons from having negative pre-activations after the first pass.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np
np.random.seed(42)

x  = np.array([0.5, -0.3, 0.8])
W1 = np.random.randn(4, 3)
b1 = np.random.randn(4)
W2 = np.random.randn(2, 4)
b2 = np.random.randn(2)

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

z1    = W1 @ x + b1
h1    = np.maximum(0, z1)
z2    = W2 @ h1 + b2
y_hat = softmax(z2)

print(f'z1:    {z1.round(3)}')    # shape (4,)
print(f'h1:    {h1.round(3)}')    # shape (4,), ReLU zeros out negatives
print(f'z2:    {z2.round(3)}')    # shape (2,)
print(f'y_hat: {y_hat.round(4)}') # shape (2,), sums to 1
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Extend the forward pass to process 5 inputs at once (batch forward pass). Change `x` to a `(5, 3)` matrix where each row is one input, and update the math to `z1 = x @ W1.T + b1`.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n\n# Batch of 5 inputs, each with 3 features\nX_batch = np.random.randn(5, 3)\n\nW1 = np.random.randn(4, 3)\nb1 = np.random.randn(4)\nW2 = np.random.randn(2, 4)\nb2 = np.random.randn(2)\n\ndef softmax_batch(Z):\n    # Apply softmax to each row\n    E = np.exp(Z - Z.max(axis=1, keepdims=True))\n    return E / E.sum(axis=1, keepdims=True)\n\n# TODO: batch forward pass\n# z1 shape should be (5, 4)\nz1 = None  # X_batch @ W1.T + b1\nh1 = None  # np.maximum(0, z1)\nz2 = None  # h1 @ W2.T + b2\ny_hat = None  # softmax_batch(z2)\n\nprint(f'X_batch shape: {X_batch.shape}')\nprint(f'z1 shape:      {z1.shape}')   # expected: (5, 4)\nprint(f'h1 shape:      {h1.shape}')   # expected: (5, 4)\nprint(f'y_hat shape:   {y_hat.shape}') # expected: (5, 2)\nprint(f'Row sums:      {y_hat.sum(axis=1).round(4)}')  # expected: [1, 1, 1, 1, 1]" height="260" >}}

2. **Coding:** Add a third hidden layer to the forward pass. Architecture: 3 → 4 → 4 → 3 → 2. Initialize `W3` and `b3`. Print all intermediate shapes.
3. **Challenge:** Why must we store intermediate values \\(z_1, h_1\\) during the forward pass? Work through the backpropagation formulas for this network and identify which intermediate values are needed for each gradient computation.
4. **Variant:** Implement the forward pass for a DQN-style network: input is a state vector of dimension 8 (like CartPole with full state), two hidden layers of 64 neurons each (ReLU), output is 2 Q-values (one per action). Use `np.random.seed(0)`. Print the Q-values and the greedy action.

{{< pyrepl code="import numpy as np\nnp.random.seed(0)\n\n# DQN-style network for CartPole: 8 inputs → 64 → 64 → 2\nstate = np.array([0.1, -0.05, 0.02, 0.3, 0.0, 0.0, 0.0, 0.0])\n\n# TODO: initialize weights (Xavier-ish: scale by 0.1)\nW1 = np.random.randn(64, 8) * 0.1\nb1 = np.zeros(64)\nW2 = np.random.randn(64, 64) * 0.1\nb2 = np.zeros(64)\nW3 = np.random.randn(2, 64) * 0.1\nb3 = np.zeros(2)\n\n# TODO: forward pass\nh1 = None  # ReLU(W1 @ state + b1)\nh2 = None  # ReLU(W2 @ h1 + b2)\nq_values = None  # W3 @ h2 + b3 (no activation for Q-values)\n\ngreedy_action = None  # np.argmax(q_values)\nprint(f'Q-values: {q_values}')\nprint(f'Greedy action: {greedy_action}')\n# expected: Q-values are small numbers near 0; action is 0 or 1" height="240" >}}

5. **Debug:** The forward pass below passes the bias as a 2D column vector instead of a 1D vector, causing a shape mismatch. Find and fix the bug.

{{< pyrepl code="import numpy as np\nnp.random.seed(1)\n\nx = np.array([0.5, -0.3, 0.8])\nW1 = np.random.randn(4, 3)\n\n# BUG: bias has wrong shape (2D instead of 1D)\nb1 = np.random.randn(4, 1)   # TODO: fix — should be shape (4,)\n\nz1 = W1 @ x + b1   # this will fail or broadcast incorrectly\nh1 = np.maximum(0, z1)\nprint(f'z1 shape: {z1.shape}')  # expected: (4,)\nprint(f'h1 shape: {h1.shape}')  # expected: (4,)" height="180" >}}

6. **Conceptual:** In a trained DQN, the hidden layer activations \\(h_1\\) and \\(h_2\\) represent learned features of the state. What might these features represent for a game like Pong? How does this differ from the raw pixel input?
7. **Recall:** Write the four equations for a 2-layer forward pass from memory. Name every variable and give its shape for a network with input dimension 3, hidden dimension 4, and output dimension 2.
