---
title: "DL Mini-Project: Digits Classifier in NumPy"
description: "Build a 2-layer MLP to classify handwritten digits using only NumPy. Full pipeline: data, init, training, evaluation."
date: 2026-03-20T00:00:00Z
weight: 13
draft: false
difficulty: 5
tags: ["mini-project", "digits classifier", "numpy", "MLP", "classification", "dl-foundations"]
keywords: ["digit classifier NumPy", "MLP from scratch", "sklearn digits", "neural network mini-project", "dl-foundations"]
roadmap_icon: "sparkles"
roadmap_color: "purple"
roadmap_phase_label: "Chapter 13 · Mini-Project"
---

**Learning objectives**
- Build a complete neural network pipeline from data loading to evaluation using only NumPy
- Implement forward pass, cross-entropy loss, backpropagation, and SGD in sequence
- Track and interpret a training loss curve
- Connect this pipeline to the DQN training pattern

**Concept and real-world motivation**

This mini-project combines everything from the DL Foundations section. You will build a 2-layer MLP to classify handwritten digits — the same pipeline used in DQN: **input → hidden layers → output**. The input is a flattened image (pixel values), the hidden layers extract features, and the output layer predicts a class (or in DQN, a Q-value per action).

We use **sklearn's digits dataset** — 1797 samples of 8×8 = 64-pixel images of digits 0–9. We take the first 100 samples to keep computation fast in the browser.

---

## Step 1 — Prepare data

{{< pyrepl code="import numpy as np\nfrom sklearn.datasets import load_digits\n\n# Load mini dataset: first 100 samples\ndigits = load_digits()\nX = digits.data[:100] / 16.0   # normalize pixel values to [0, 1]\ny = digits.target[:100]\nprint('X shape:', X.shape, '  (100 samples, 64 features)')\nprint('y shape:', y.shape, '  classes:', np.unique(y))\n\n# One-hot encode labels\ndef one_hot(y, n_classes=10):\n    Y = np.zeros((len(y), n_classes))\n    Y[np.arange(len(y)), y] = 1\n    return Y\nY = one_hot(y)\nprint('Y shape:', Y.shape, '  (one-hot encoded)')\n\n# Train/test split (80/20)\nnp.random.seed(42)\nidx = np.random.permutation(100)\ntrain_idx, test_idx = idx[:80], idx[80:]\nX_train, X_test = X[train_idx], X[test_idx]\nY_train, Y_test = Y[train_idx], Y[test_idx]\ny_test = y[test_idx]\nprint('\\nTrain size:', len(X_train), '  Test size:', len(X_test))" height="300" >}}

---

## Step 2 — Initialize the MLP

Architecture: **64 → 32 → 10** (input features → hidden → output classes)

{{< pyrepl code="import numpy as np\n\n# Run Step 1 first to get X_train, Y_train, X_test, Y_test\nfrom sklearn.datasets import load_digits\ndigits = load_digits()\nX = digits.data[:100] / 16.0\ny = digits.target[:100]\nY = np.zeros((100, 10)); Y[np.arange(100), y] = 1\nnp.random.seed(42)\nidx = np.random.permutation(100)\nX_train, X_test = X[idx[:80]], X[idx[80:]]\nY_train, Y_test = Y[idx[:80]], Y[idx[80:]]\ny_test = y[idx[80:]]\n\n# Initialize MLP: 64 -> 32 -> 10\nnp.random.seed(42)\nW1 = np.random.randn(32, 64) * 0.01\nb1 = np.zeros(32)\nW2 = np.random.randn(10, 32) * 0.01\nb2 = np.zeros(10)\nprint('W1 shape:', W1.shape)\nprint('W2 shape:', W2.shape)\nprint('Total parameters:', W1.size + b1.size + W2.size + b2.size)" height="240" >}}

---

## Step 3 — Training loop

{{< pyrepl code="import numpy as np\nfrom sklearn.datasets import load_digits\n\n# Setup (combining steps 1 and 2)\ndigits = load_digits()\nX = digits.data[:100] / 16.0\ny = digits.target[:100]\nY = np.zeros((100, 10)); Y[np.arange(100), y] = 1\nnp.random.seed(42)\nidx = np.random.permutation(100)\nX_train, X_test = X[idx[:80]], X[idx[80:]]\nY_train, Y_test = Y[idx[:80]], Y[idx[80:]]\ny_test = y[idx[80:]]\nnp.random.seed(42)\nW1 = np.random.randn(32, 64) * 0.01; b1 = np.zeros(32)\nW2 = np.random.randn(10, 32) * 0.01; b2 = np.zeros(10)\n\ndef relu(z): return np.maximum(0, z)\ndef softmax(z):\n    e = np.exp(z - z.max(axis=1, keepdims=True))\n    return e / e.sum(axis=1, keepdims=True)\n\nlr = 0.1\nlosses = []\nfor epoch in range(200):\n    # Shuffle training data\n    p = np.random.permutation(len(X_train))\n    Xb, Yb = X_train[p], Y_train[p]\n    # Forward pass\n    z1 = Xb @ W1.T + b1\n    a1 = relu(z1)\n    z2 = a1 @ W2.T + b2\n    a2 = softmax(z2)\n    # Cross-entropy loss\n    loss = -np.mean(np.sum(Yb * np.log(a2 + 1e-9), axis=1))\n    losses.append(loss)\n    # Backprop\n    d2 = (a2 - Yb) / len(Xb)\n    dW2 = d2.T @ a1; db2 = d2.sum(0)\n    d1 = (d2 @ W2) * (z1 > 0)\n    dW1 = d1.T @ Xb; db1 = d1.sum(0)\n    # SGD update\n    W2 -= lr * dW2; b2 -= lr * db2\n    W1 -= lr * dW1; b1 -= lr * db1\n    if (epoch + 1) % 40 == 0:\n        print(f'Epoch {epoch+1}: loss = {loss:.4f}')" height="380" >}}

---

## Step 4 — Plot loss curve

{{< pyrepl code="import numpy as np\nimport matplotlib.pyplot as plt\n\n# Assumes 'losses' list from Step 3\n# If running fresh, re-run Step 3 first, then this cell\n# For demonstration, generate a synthetic loss curve\nnp.random.seed(42)\nlosses_demo = 2.3 * np.exp(-np.arange(200) * 0.02) + 0.05 * np.random.randn(200) * np.exp(-np.arange(200)*0.01)\nlosses_demo = np.maximum(losses_demo, 0.05)\n\ntry:\n    plot_losses = losses  # from Step 3\nexcept NameError:\n    plot_losses = losses_demo  # fallback demo\n\nplt.figure(figsize=(8, 4))\nplt.plot(plot_losses, color='steelblue', linewidth=1.5)\nplt.xlabel('Epoch')\nplt.ylabel('Cross-entropy loss')\nplt.title('Training loss curve')\nplt.grid(alpha=0.3)\nplt.tight_layout()\nplt.savefig('loss_curve.png', dpi=80)\nplt.show()\nprint(f'Final loss: {plot_losses[-1]:.4f}')" height="260" >}}

---

## Step 5 — Evaluate on test set

{{< pyrepl code="import numpy as np\nfrom sklearn.datasets import load_digits\n\n# Full pipeline (re-run to get trained weights)\ndigits = load_digits()\nX = digits.data[:100] / 16.0\ny = digits.target[:100]\nY = np.zeros((100, 10)); Y[np.arange(100), y] = 1\nnp.random.seed(42)\nidx = np.random.permutation(100)\nX_train, X_test = X[idx[:80]], X[idx[80:]]\nY_train, Y_test = Y[idx[:80]], Y[idx[80:]]\ny_test = y[idx[80:]]\nnp.random.seed(42)\nW1 = np.random.randn(32, 64) * 0.01; b1 = np.zeros(32)\nW2 = np.random.randn(10, 32) * 0.01; b2 = np.zeros(10)\nrelu = lambda z: np.maximum(0, z)\ndef softmax(z):\n    e = np.exp(z - z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)\nlr = 0.1\nfor epoch in range(200):\n    p = np.random.permutation(len(X_train))\n    Xb, Yb = X_train[p], Y_train[p]\n    z1 = Xb @ W1.T + b1; a1 = relu(z1)\n    z2 = a1 @ W2.T + b2; a2 = softmax(z2)\n    d2 = (a2 - Yb) / len(Xb)\n    dW2 = d2.T @ a1; db2 = d2.sum(0)\n    d1 = (d2 @ W2) * (z1 > 0)\n    dW1 = d1.T @ Xb; db1 = d1.sum(0)\n    W2 -= lr * dW2; b2 -= lr * db2\n    W1 -= lr * dW1; b1 -= lr * db1\n# Evaluate\nz1_test = X_test @ W1.T + b1; a1_test = relu(z1_test)\nz2_test = a1_test @ W2.T + b2; a2_test = softmax(z2_test)\ny_pred = a2_test.argmax(axis=1)\nacc = np.mean(y_pred == y_test)\nprint('Test accuracy:', f'{acc*100:.1f}%')\nprint('Predictions:', y_pred)\nprint('True labels:', y_test)" height="220" >}}

---

**Debug exercise:** Fix the softmax that doesn't sum to 1 (missing normalization):

{{< pyrepl code="import numpy as np\n\n# BUG: softmax missing normalization (sum of exp, not sum of probabilities)\ndef softmax_buggy(z):\n    e = np.exp(z - z.max())  # subtract max for numerical stability\n    return e  # BUG: forgot to divide by sum\n\nlogits = np.array([2.0, 1.0, 0.5])\nprobs = softmax_buggy(logits)\nprint('Buggy output:', probs)\nprint('Sum (should be 1.0):', probs.sum())\n# TODO: fix softmax_buggy so output sums to 1" height="180" >}}

**Professor's hints**
- On only 80 training samples, the network can memorize the data. Watch the loss curve — if it goes to near-zero, the model is overfitting on this tiny dataset.
- With `lr=0.1` on 200 epochs you should see clear learning. If loss barely moves, try `lr=0.5`.
- The test accuracy with 100 samples and simple MLP will be modest (~50–70%) — this is expected. With all 1797 samples, it reaches ~95%.

**Common pitfalls**
- Running the evaluation cell without first running the training cell (weights won't be trained).
- Using the wrong axis in softmax: use `axis=1` for batches (rows are samples), not `axis=0`.

{{< collapse summary="Worked solution comparison with PyTorch" >}}
For a PyTorch comparison, use the local notebook:
{{< /collapse >}}

{{< notebook path="dl-foundations/dl_mini_project.ipynb" title="DL Mini-Project in PyTorch (run locally)" >}}

**Extra practice**

1. **Warm-up:** Run only Step 1. Print the pixel values of the first training sample. Reshape it to 8×8 and print.

2. **Coding:** Add L2 regularization (lambda=0.01) to the training loop in Step 3. Does the test accuracy improve?

3. **Challenge:** Scale to all 1797 samples. Add a third hidden layer (64→128→64→10). What test accuracy do you achieve?

4. **Variant:** Replace SGD with a hand-coded Adam optimizer in the training loop. Compare convergence speed.

5. **Debug:** Modify Step 3 to introduce a bug: divide by `n_classes` instead of `len(Xb)` in the gradient. Observe how training is affected.

6. **Conceptual:** How does this digits classifier pipeline compare to DQN? Map: input → state, hidden layers → feature extraction, output → Q-values/actions.

7. **Recall:** In 3 steps, describe the full training pipeline you implemented from raw pixels to accuracy score.
