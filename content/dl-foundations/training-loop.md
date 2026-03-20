---
title: "The Training Loop"
description: "Build a full training loop in NumPy: batches, epochs, forward pass, backprop, and weight updates."
date: 2026-03-20T00:00:00Z
weight: 9
draft: false
difficulty: 5
tags: ["deep learning", "training loop", "batches", "epochs", "backpropagation", "dl-foundations"]
keywords: ["training loop", "mini-batch", "epochs", "neural network training", "NumPy MLP", "forward pass", "backprop"]
roadmap_icon: "terminal"
roadmap_color: "blue"
roadmap_phase_label: "Chapter 9"
---

**Learning objectives**
- Implement a complete training loop: forward pass → loss → backprop → weight update
- Understand the role of mini-batches and epochs in training efficiency
- Track loss over epochs and interpret a learning curve
- Connect the training loop pattern to DQN's replay buffer training

**Concept and real-world motivation**

**Training** a neural network means repeatedly: (1) run a forward pass to get predictions, (2) compute the loss, (3) run backpropagation to get gradients, (4) update weights using an optimizer. This loop runs for many **epochs** (full passes over the training data). Each epoch is divided into **mini-batches** — subsets of the data processed together.

Why mini-batches? Computing gradients on one sample at a time (SGD) is noisy but fast per step. Computing on the whole dataset is stable but slow. Mini-batches balance these: enough samples for a stable gradient estimate, processed efficiently in parallel.

**In RL:** The DQN training loop samples a mini-batch from the replay buffer, does a forward pass to compute Q-values, computes the TD loss (a form of MSE), runs backprop through the Q-network, and updates with Adam. The "replay buffer" plays the role of the training dataset. The key difference from supervised learning: the targets change as the network improves — this instability is why DQN needs a target network.

**Illustration — Training loop flow:**

```
┌─────────────────────────────────────────────────────────┐
│  Initialize weights                                      │
│         ↓                                               │
│  For each epoch:                                        │
│    For each mini-batch:                                 │
│      → Forward pass (compute predictions)               │
│      → Compute loss                                     │
│      → Backpropagation (compute gradients)              │
│      → Update weights (SGD / Adam)                      │
│         ↓                                               │
│  Log loss every N epochs → learning curve               │
└─────────────────────────────────────────────────────────┘
```

**Exercise:** Complete the full training loop for a 2-layer MLP on synthetic binary classification data.

{{< pyrepl code="import numpy as np\n\nnp.random.seed(42)\n# Generate 100 samples, 2 features, 2 classes\nn = 100\nX = np.random.randn(n, 2)\ny = (X[:, 0] + X[:, 1] > 0).astype(int)  # class = 1 if x0+x1 > 0\n\n# One-hot encode\ndef one_hot(y, n_classes=2):\n    Y = np.zeros((len(y), n_classes))\n    Y[np.arange(len(y)), y] = 1\n    return Y\nY = one_hot(y)\n\n# Initialize weights: 2->4->2 network\nW1 = np.random.randn(4, 2) * 0.1\nb1 = np.zeros(4)\nW2 = np.random.randn(2, 4) * 0.1\nb2 = np.zeros(2)\nlr = 0.05\n\ndef relu(z): return np.maximum(0, z)\ndef softmax(z):\n    e = np.exp(z - z.max(axis=1, keepdims=True))\n    return e / e.sum(axis=1, keepdims=True)\n\nlosses = []\nepochs = 50\nbatch_size = 10\n\nfor epoch in range(epochs):\n    idx = np.random.permutation(n)\n    epoch_loss = 0\n    for i in range(0, n, batch_size):\n        xb = X[idx[i:i+batch_size]]\n        yb = Y[idx[i:i+batch_size]]\n        # Forward pass\n        z1 = xb @ W1.T + b1\n        a1 = relu(z1)\n        z2 = a1 @ W2.T + b2\n        a2 = softmax(z2)\n        # Cross-entropy loss\n        loss = -np.mean(np.sum(yb * np.log(a2 + 1e-9), axis=1))\n        epoch_loss += loss\n        # Backprop\n        delta2 = (a2 - yb) / len(xb)\n        dW2 = delta2.T @ a1\n        db2 = delta2.sum(axis=0)\n        delta1 = (delta2 @ W2) * (z1 > 0)\n        dW1 = delta1.T @ xb\n        db1 = delta1.sum(axis=0)\n        # SGD update\n        W2 -= lr * dW2\n        b2 -= lr * db2\n        W1 -= lr * dW1\n        b1 -= lr * db1\n    losses.append(epoch_loss)\n    if (epoch + 1) % 10 == 0:\n        print(f'Epoch {epoch+1:3d}: loss = {epoch_loss:.4f}')" height="400" >}}

**Professor's hints**
- Always shuffle the data at the start of each epoch (`np.random.permutation`) to avoid the network seeing the same order every time.
- The learning curve should generally decrease. If it goes up after initially going down, the learning rate may be too large.
- The backward pass mirrors the forward pass in reverse order — backprop through the last layer first.
- Numerical stability: always add a small \\(\epsilon\\) (like `1e-9`) inside `log()` to avoid `log(0) = -inf`.

**Common pitfalls**
- Forgetting to subtract the gradient (adding it instead makes the loss increase).
- Not zeroing out momentum/Adam state between epochs if you store it outside the loop.
- Using the wrong denominator for the gradient: divide by batch size for the mean loss.

**Extra practice**

1. **Warm-up:** Implement forward pass + cross-entropy loss for one batch only, without the training loop.
{{< pyrepl code="import numpy as np\nnp.random.seed(0)\n# One batch: 4 samples, 2 features, 2 classes\nX_batch = np.random.randn(4, 2)\ny_batch = np.array([[1,0],[0,1],[1,0],[0,1]])\n# Initialize small network\nW1 = np.random.randn(3, 2) * 0.1\nb1 = np.zeros(3)\nW2 = np.random.randn(2, 3) * 0.1\nb2 = np.zeros(2)\n# TODO: forward pass + cross-entropy loss\n# expected: loss between 0.5 and 1.5 for random init\nrelu = lambda z: np.maximum(0, z)\ndef softmax(z):\n    e = np.exp(z - z.max(axis=1, keepdims=True))\n    return e / e.sum(axis=1, keepdims=True)\nz1 = X_batch @ W1.T + b1\na1 = relu(z1)\nz2 = a1 @ W2.T + b2\na2 = softmax(z2)\nloss = -np.mean(np.sum(y_batch * np.log(a2 + 1e-9), axis=1))\nprint('loss:', round(loss, 4))" height="280" >}}

2. **Coding:** Add accuracy tracking to the training loop above: after each epoch, compute training accuracy (percentage of correct predictions).

3. **Challenge:** Modify the training loop to use Adam instead of SGD. Track the loss and compare learning curves.

4. **Variant:** Implement learning rate decay: multiply `lr` by 0.95 after each epoch. Observe the effect on convergence.

5. **Debug:** Fix the training loop where weights are not being updated (gradient is added instead of subtracted):
{{< pyrepl code="import numpy as np\nnp.random.seed(0)\n# Simple 1-layer: minimize MSE on linear data\nX = np.array([[1.0],[2.0],[3.0],[4.0]])\ny = np.array([2.0, 4.0, 6.0, 8.0])\nW = np.array([[0.1]])\nb = np.array([0.0])\nlr = 0.01\nfor epoch in range(50):\n    pred = X @ W.T + b\n    loss = np.mean((pred.flatten() - y)**2)\n    grad_W = 2 * np.mean((pred.flatten() - y)[:, None] * X, axis=0)\n    grad_b = 2 * np.mean(pred.flatten() - y)\n    W = W + lr * grad_W  # BUG: should be -= not +=\n    b = b + lr * grad_b  # BUG: should be -= not +=\nprint('W (buggy):', round(W[0,0], 4), '  expected ~2.0')\n# TODO: fix the bugs" height="260" >}}

6. **Conceptual:** In DQN, the "training data" changes over time as the agent collects new experience. How does the replay buffer address this? Why can't you just use the last transition as a single-sample batch?

7. **Recall:** What is the difference between a batch, a mini-batch, and an epoch? Why do we shuffle data at the start of each epoch?
