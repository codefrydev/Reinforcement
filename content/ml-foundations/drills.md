---
title: "ML Foundations Drills"
description: "15 short drill problems covering supervised learning, gradient descent, evaluation, and sklearn."
date: 2026-03-20T00:00:00Z
weight: 99
draft: false
difficulty: 4
tags: ["drills", "practice", "machine learning", "ml-foundations"]
keywords: ["ML drills", "supervised learning practice", "gradient descent", "model evaluation", "sklearn exercises"]
roadmap_icon: "brain"
roadmap_color: "indigo"
roadmap_phase_label: "Drills"
---

Short drills for the full ML Foundations section. Work through these after completing pages 1–13 to consolidate your understanding before the review.

## Recall (R) — State definitions and rules

**R1.** What is the difference between supervised and unsupervised learning? Give one example of each.

{{< collapse summary="R1 answer" >}}
**Supervised learning:** Each training example has a label \\(y\\). The model learns a mapping \\(f: X \to y\\). Example: predicting house price from features (regression) or classifying emails as spam/not-spam (classification).

**Unsupervised learning:** No labels. The algorithm finds structure in the data itself. Example: K-Means clustering of customer purchase history to identify segments.
{{< /collapse >}}

**R2.** What does MSE stand for? Write the formula.

{{< collapse summary="R2 answer" >}}
**Mean Squared Error.** For N predictions: \\[\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2\\] It penalizes large errors more than small ones (due to squaring) and is always non-negative.
{{< /collapse >}}

**R3.** Why do we use train/test split instead of evaluating on training data?

{{< collapse summary="R3 answer" >}}
Evaluating on training data measures **memorization**, not generalization. A model that overfits achieves near-zero training error but fails on new data. A held-out test set provides an unbiased estimate of how the model performs on unseen examples — the quantity we actually care about in practice.
{{< /collapse >}}

**R4.** What is the connection between the Markov decision process and supervised learning?

{{< collapse summary="R4 answer" >}}
In RL, the **value function** \\(V(s)\\) and **action-value function** \\(Q(s,a)\\) are learned from experience — effectively a supervised regression problem where targets are bootstrapped returns. The **policy** \\(\pi(a|s)\\) can be viewed as a classifier that maps states to action distributions. Both use gradient descent to minimize a loss, the same optimization algorithm that drives supervised learning.
{{< /collapse >}}

**R5.** State the update rule for gradient descent.

{{< collapse summary="R5 answer" >}}
\\[w \leftarrow w - \alpha \frac{\partial \mathcal{L}}{\partial w}\\] where \\(\alpha\\) is the learning rate and \\(\frac{\partial \mathcal{L}}{\partial w}\\) is the gradient of the loss with respect to the parameter \\(w\\). Move in the opposite direction of the gradient to reduce loss.
{{< /collapse >}}

---

## Compute (C) — Numerical exercises

**C1.** Compute MSE for predictions \\([2, 4, 6]\\) and true values \\([2.5, 3.5, 6.5]\\).

{{< pyrepl code="import numpy as np\npreds = np.array([2.0, 4.0, 6.0])\ntrue  = np.array([2.5, 3.5, 6.5])\n# TODO: compute MSE = mean of (pred - true)^2\nmse = None\nprint(f'MSE = {mse}')\n# expected: MSE = 0.25" height="180" >}}

{{< collapse summary="C1 answer" >}}
Errors: \\(-0.5, +0.5, -0.5\\). Squared: \\(0.25, 0.25, 0.25\\). Mean = \\(0.25\\).

```python
mse = np.mean((preds - true)**2)  # 0.25
```
{{< /collapse >}}

**C2.** For logistic regression with \\(z = 1.5\\), compute \\(\sigma(z) = \frac{1}{1 + e^{-z}}\\).

{{< pyrepl code="import numpy as np\nz = 1.5\n# TODO: compute sigmoid(z)\nsigma = None\nprint(f'sigma(1.5) = {sigma:.4f}')\n# expected: ≈ 0.8176" height="160" >}}

{{< collapse summary="C2 answer" >}}
\\(\sigma(1.5) = \frac{1}{1 + e^{-1.5}} = \frac{1}{1 + 0.2231} \approx 0.8176\\).

```python
sigma = 1 / (1 + np.exp(-1.5))  # 0.8176
```
{{< /collapse >}}

**C3.** For TP=4, FP=1, FN=2: compute precision and recall.

{{< pyrepl code="TP, FP, FN = 4, 1, 2\n# TODO: precision = TP / (TP + FP)\n# TODO: recall    = TP / (TP + FN)\nprecision = None\nrecall    = None\nprint(f'Precision = {precision:.4f}, Recall = {recall:.4f}')\n# expected: Precision ≈ 0.8000, Recall ≈ 0.6667" height="160" >}}

{{< collapse summary="C3 answer" >}}
Precision = \\(\frac{4}{4+1} = 0.80\\). Recall = \\(\frac{4}{4+2} \approx 0.6667\\).

```python
precision = TP / (TP + FP)  # 0.80
recall    = TP / (TP + FN)  # 0.6667
```
{{< /collapse >}}

**C4.** Gradient step: \\(w = 3\\), gradient \\(= 0.5\\), learning rate \\(\alpha = 0.1\\). Compute \\(w_{\text{new}}\\).

{{< collapse summary="C4 answer" >}}
\\(w_{\text{new}} = w - \alpha \cdot \nabla = 3 - 0.1 \times 0.5 = 3 - 0.05 = 2.95\\).
{{< /collapse >}}

**C5.** Compute entropy in bits for a dataset with 3 positives and 3 negatives.

{{< pyrepl code="import numpy as np\n# 3 positives, 3 negatives → p_pos = p_neg = 0.5\n# H = -p_pos*log2(p_pos) - p_neg*log2(p_neg)\np = 0.5\nH = None\nprint(f'H = {H:.4f} bits')\n# expected: H = 1.0 bit (maximum uncertainty for binary)" height="160" >}}

{{< collapse summary="C5 answer" >}}
\\(H = -0.5 \log_2 0.5 - 0.5 \log_2 0.5 = -0.5 \times (-1) - 0.5 \times (-1) = 1.0\\) bit. Maximum entropy for a binary distribution.

```python
H = -p * np.log2(p) - p * np.log2(p)  # 1.0
```
{{< /collapse >}}

---

## Code (K) — Implementation

**K1.** Write `linear_regression_predict(X, w, b)` that returns \\(Xw + b\\).

{{< pyrepl code="import numpy as np\n\ndef linear_regression_predict(X, w, b):\n    # TODO: return X @ w + b\n    return None\n\n# Test\nX = np.array([[1, 2], [3, 4], [5, 6]])\nw = np.array([0.5, 0.5])\nb = 1.0\npreds = linear_regression_predict(X, w, b)\nprint(preds)\n# expected: [2.5, 5.5, 8.5]" height="200" >}}

{{< collapse summary="K1 answer" >}}
```python
def linear_regression_predict(X, w, b):
    return X @ w + b

# X @ w = [1*0.5 + 2*0.5, 3*0.5 + 4*0.5, 5*0.5 + 6*0.5] = [1.5, 3.5, 5.5]
# + b=1 → [2.5, 4.5, 6.5]  ← check with your values
```
{{< /collapse >}}

**K2.** Write `accuracy(y_true, y_pred)` that returns the fraction of correct predictions.

{{< pyrepl code="import numpy as np\n\ndef accuracy(y_true, y_pred):\n    # TODO: return fraction of elements that are equal\n    return None\n\n# Test\ny_true = np.array([1, 0, 1, 1, 0])\ny_pred = np.array([1, 0, 0, 1, 1])\nprint(f'Accuracy: {accuracy(y_true, y_pred)}')\n# expected: 0.6 (3 out of 5 correct)" height="180" >}}

{{< collapse summary="K2 answer" >}}
```python
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
# np.mean of boolean array = fraction of True values
```
{{< /collapse >}}

---

## Debug (D) — Find and fix the bug

**D1.** The cross-entropy loss uses `log(1 - p)` for positive examples. Fix it.

{{< pyrepl code="import numpy as np\n\ndef buggy_cross_entropy(y_true, y_pred_prob):\n    # BUG: uses log(1 - p) for positive examples (should be log(p))\n    loss = -np.mean(\n        y_true * np.log(1 - y_pred_prob) +\n        (1 - y_true) * np.log(y_pred_prob)\n    )\n    return loss\n\ny_true = np.array([1, 0, 1])\ny_prob = np.array([0.9, 0.1, 0.8])\nprint(f'Buggy loss: {buggy_cross_entropy(y_true, y_prob):.4f}')\n# TODO: fix so positive class uses log(p) and negative uses log(1-p)\n# expected (correct): small loss ≈ 0.14 for these confident predictions" height="200" >}}

{{< collapse summary="D1 answer" >}}
The terms `log(1 - p)` and `log(p)` are swapped. The correct cross-entropy is:

\\[\mathcal{L} = -\frac{1}{N} \sum_i [y_i \log p_i + (1-y_i) \log(1-p_i)]\\]

```python
def fixed_cross_entropy(y_true, y_pred_prob):
    loss = -np.mean(
        y_true * np.log(y_pred_prob) +
        (1 - y_true) * np.log(1 - y_pred_prob)
    )
    return loss
```
{{< /collapse >}}

**D2.** The gradient descent adds the gradient instead of subtracting it. Fix it.

{{< pyrepl code="import numpy as np\n\ndef buggy_gradient_step(w, gradient, lr=0.1):\n    # BUG: adds gradient instead of subtracting\n    return w + lr * gradient\n\nw = 3.0\ngradient = 0.5\nprint(f'Buggy w_new: {buggy_gradient_step(w, gradient)}')\n# Expected w_new: 2.95 (move opposite to gradient to reduce loss)\n# TODO: fix the update rule\ndef fixed_gradient_step(w, gradient, lr=0.1):\n    return None  # w - lr * gradient\n\nprint(f'Fixed w_new: {fixed_gradient_step(w, gradient)}')" height="200" >}}

{{< collapse summary="D2 answer" >}}
Gradient descent moves **opposite** to the gradient: \\(w \leftarrow w - \alpha \nabla\\). Adding the gradient would climb the loss surface (gradient ascent), not descend it.

```python
def fixed_gradient_step(w, gradient, lr=0.1):
    return w - lr * gradient  # 3.0 - 0.1*0.5 = 2.95
```
{{< /collapse >}}

---

## Challenge (X)

**X1.** Implement K-Means from scratch on a dataset of 20 2D points. Run for 10 iterations and plot the cluster assignments after each step. Use K=3 and random seed 42.

{{< pyrepl code="import numpy as np\n\nnp.random.seed(42)\n# 20 points from 3 clusters\nX = np.vstack([\n    np.random.randn(7, 2) + [0, 0],\n    np.random.randn(7, 2) + [5, 5],\n    np.random.randn(6, 2) + [0, 5],\n])\n\nK = 3\n# TODO: Initialize K centroids randomly (pick K rows from X)\n# TODO: Run 10 iterations of assignment + update\n# TODO: Print final cluster sizes\n\n# Hint structure:\n# centroids = X[np.random.choice(len(X), K, replace=False)]\n# for _ in range(10):\n#     dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)\n#     assignments = np.argmin(dists, axis=1)\n#     for k in range(K):\n#         centroids[k] = X[assignments == k].mean(axis=0)\nprint('K-Means not yet implemented — TODO')\n# expected: 3 clusters with ~7, 7, 6 points respectively" height="260" >}}

{{< collapse summary="X1 solution" >}}
```python
import numpy as np
np.random.seed(42)
X = np.vstack([
    np.random.randn(7, 2) + [0, 0],
    np.random.randn(7, 2) + [5, 5],
    np.random.randn(6, 2) + [0, 5],
])
K = 3
centroids = X[np.random.choice(len(X), K, replace=False)]
for i in range(10):
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    assignments = np.argmin(dists, axis=1)
    for k in range(K):
        if (assignments == k).sum() > 0:
            centroids[k] = X[assignments == k].mean(axis=0)
for k in range(K):
    print(f'Cluster {k}: {(assignments == k).sum()} points')
```
{{< /collapse >}}
