---
title: "Phase 4 Assessment: Machine Learning Foundations"
description: "12 questions covering supervised learning, gradient descent, model evaluation, and sklearn. Pass: 9/12."
date: 2026-03-20T00:00:00Z
draft: false
tags: ["assessment", "phase 4", "machine learning", "ml-foundations", "self-check", "solutions"]
keywords: ["phase 4 assessment", "machine learning quiz", "ML foundations check"]
weight: 5
roadmap_icon: "trend-up"
roadmap_color: "amber"
roadmap_phase_label: "Phase 4 Quiz"
---

Use this self-check after completing [ML Foundations](../ml-foundations/). Pass: **9 out of 12**. If you score below 9, review the topics you missed before continuing to Phase 5 (DL Foundations).

---

### 1. Predict the output

Which category does each problem belong to: supervised learning, unsupervised learning, or reinforcement learning?

- (a) Predicting house prices from square footage and location.
- (b) Grouping news articles by topic without any pre-defined categories.
- (c) Teaching a robot to walk by giving it +1 for each step it takes without falling.

{{< collapse summary="Answer" >}}
(a) **Supervised** — you have labeled examples (house price = label, features = input).

(b) **Unsupervised** — no labels; the algorithm discovers clusters.

(c) **Reinforcement learning** — the agent receives a reward signal and must learn from interaction.
{{< /collapse >}}

---

### 2. Write a function

Implement MSE (mean squared error) for two arrays.

{{< pyrepl code="import numpy as np\n\ndef mse(y_true, y_pred):\n    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)\n\nprint(mse([1, 2, 3], [1, 2, 3]))    # 0.0\nprint(mse([1, 2, 3], [2, 3, 4]))    # 1.0\nprint(mse([0, 0], [1, 1]))           # 1.0" height="180" >}}

{{< collapse summary="Answer" >}}
```python
def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
```
MSE = average of squared differences. It penalizes large errors more than small ones (squared). Used in linear regression and DQN's TD loss.
{{< /collapse >}}

---

### 3. Find the bug

This gradient descent loop is supposed to minimize `w` toward the minimum of `f(w) = (w-5)^2`, but `w` diverges instead of converging.

```python
w = 0.0
lr = 0.1
for _ in range(50):
    gradient = 2 * (w - 5)
    w = w + lr * gradient   # bug
```

{{< collapse summary="Answer" >}}
Bug: `w = w + lr * gradient` **adds** the gradient instead of subtracting it. Gradient descent moves **opposite** to the gradient to minimize the loss.

Fix: `w = w - lr * gradient`

With the fix, \\(w\\) converges to 5 (the minimum of \\((w-5)^2\\)). With the bug, \\(w\\) moves away from 5 and diverges.
{{< /collapse >}}

---

### 4. Predict the output

For a logistic regression model, what are `sigmoid(0)`, `sigmoid(1)`, and `sigmoid(-1)`?

{{< collapse summary="Answer" >}}
- `sigmoid(0) = 0.5` (the decision boundary — equal probability for both classes)
- `sigmoid(1) ≈ 0.731` (more likely class 1)
- `sigmoid(-1) ≈ 0.269` (more likely class 0)

These outputs are interpreted as probabilities. If `sigmoid(z) > 0.5`, predict class 1; otherwise predict class 0.
{{< /collapse >}}

---

### 5. Write a function

Implement accuracy: fraction of predictions matching true labels.

{{< pyrepl code="import numpy as np\n\ndef accuracy(y_true, y_pred):\n    return np.mean(np.array(y_true) == np.array(y_pred))\n\nprint(accuracy([0,1,1,0], [0,1,1,0]))   # 1.0\nprint(accuracy([0,1,1,0], [0,0,1,1]))   # 0.5\nprint(accuracy([1,1,1], [0,0,0]))        # 0.0" height="160" >}}

{{< collapse summary="Answer" >}}
```python
def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))
```
`==` returns a boolean array (True/False = 1/0), and `np.mean` computes the fraction of True values.
{{< /collapse >}}

---

### 6. Find the bug

This preprocessing pipeline has a data leakage bug. Find and fix it.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X)           # bug: fitted on all data including test set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

{{< collapse summary="Answer" >}}
Bug: `scaler.fit(X)` fits on the **entire** dataset including test samples. This leaks test information (mean and std of test data) into the preprocessing, giving optimistic results that won't generalize to truly unseen data.

Fix:
```python
scaler = StandardScaler()
scaler.fit(X_train)          # fit ONLY on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)   # transform test using train statistics
```
{{< /collapse >}}

---

### 7. Conceptual

What is overfitting? How does cross-validation help detect and prevent it?

{{< collapse summary="Answer" >}}
**Overfitting** occurs when a model memorizes the training data — it performs well on training examples but poorly on new, unseen data. Signs: training accuracy >> validation accuracy; training loss << validation loss.

**Cross-validation** (e.g. k-fold) helps by: (1) **detecting** overfitting — if CV score << training score, the model is overfitting; (2) providing a more reliable estimate of generalization performance by testing on multiple non-overlapping validation sets; (3) guiding **hyperparameter selection** without touching the held-out test set.
{{< /collapse >}}

---

### 8. Predict the output

K-nearest neighbors (K=1) with training points: A=(1,1,class 0), B=(3,3,class 1), C=(2,2,class 0). For new point P=(2.5, 2.5), what does K=1 KNN predict?

{{< collapse summary="Answer" >}}
Compute distances from P=(2.5,2.5) to each training point:
- d(P,A) = sqrt((2.5-1)²+(2.5-1)²) = sqrt(4.5) ≈ 2.12
- d(P,B) = sqrt((2.5-3)²+(2.5-3)²) = sqrt(0.5) ≈ 0.71
- d(P,C) = sqrt((2.5-2)²+(2.5-2)²) = sqrt(0.5) ≈ 0.71

B and C are tied. In case of ties, scikit-learn picks the first in training order. Here the **closest point is B (class 1)** by a strict 0-distance tie-break or by convention. Prediction: **class 1** (if B wins) or class 0 (if C wins).

In practice, use K>1 to avoid sensitivity to single-point ties.
{{< /collapse >}}

---

### 9. Write a function

Compute precision given TP, FP, FN counts.

{{< pyrepl code="def precision(TP, FP, FN):\n    # precision = TP / (TP + FP)\n    return TP / (TP + FP) if (TP + FP) > 0 else 0.0\n\nprint(precision(TP=4, FP=1, FN=2))  # expected: 0.8\nprint(precision(TP=0, FP=5, FN=3))  # expected: 0.0\nprint(precision(TP=3, FP=0, FN=1))  # expected: 1.0" height="160" >}}

{{< collapse summary="Answer" >}}
```python
def precision(TP, FP, FN):
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0
```
Precision = TP / (TP + FP) = "of all predicted positives, what fraction are actually positive?" FN is not used for precision (it's used for recall = TP / (TP + FN)).
{{< /collapse >}}

---

### 10. Find the bug

This entropy formula is used in a decision tree. Find the bug.

```python
import numpy as np

def entropy(p):
    # Binary entropy: -p*log(p) - (1-p)*log(1-p)
    return -(p * np.log(p) + (1-p) * np.log(1-p))   # bug
```

{{< collapse summary="Answer" >}}
Bug: `np.log` computes natural log (base \\(e\\)). Decision tree information gain uses **log base 2** so entropy is measured in **bits**.

Fix: replace `np.log` with `np.log2`:
```python
def entropy(p):
    if p == 0 or p == 1: return 0.0  # also handle edge cases
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))
```
With `log2`, entropy(0.5) = 1.0 bit (maximum uncertainty for binary). With `log`, entropy(0.5) ≈ 0.693 (nats, not bits).
{{< /collapse >}}

---

### 11. Predict the output

After training a logistic regression classifier, what does `model.predict_proba(X)` return? What is the shape of the output for 5 samples and 3 classes?

{{< collapse summary="Answer" >}}
`predict_proba` returns a 2D array of shape **(n_samples, n_classes)** — for 5 samples and 3 classes, shape is **(5, 3)**.

Each row contains the predicted probability for each class: values are in [0,1] and each row sums to 1. Row \\(i\\) gives \\([P(\text{class 0}|x_i), P(\text{class 1}|x_i), P(\text{class 2}|x_i)]\\).

To get hard predictions: `np.argmax(predict_proba(X), axis=1)` (same as `predict(X)`).
{{< /collapse >}}

---

### 12. Conceptual

Why does linear regression fail on XOR? What do we need instead?

{{< collapse summary="Answer" >}}
**XOR** is not linearly separable: no single straight line can separate the two classes (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0. Linear regression / logistic regression can only learn linear decision boundaries.

To solve XOR, we need **non-linear models**: a neural network with at least one hidden layer (even 2 hidden neurons suffice), or kernelized SVMs, or decision trees. The hidden layer learns non-linear feature combinations that make the problem linearly separable in the transformed space.

**In RL context:** Q-tables are linear in state; for complex state spaces (e.g. continuous or high-dimensional), we need non-linear function approximators (neural networks). XOR is the classic demonstration of why.
{{< /collapse >}}

---

**Score:** 9–12: Ready for Phase 5 (DL Foundations). 7–8: Review the specific topics you missed before continuing. Below 7: Complete [ML Foundations](../ml-foundations/) and return to this assessment.
