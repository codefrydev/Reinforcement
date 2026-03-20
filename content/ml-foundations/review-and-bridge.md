---
title: "ML Foundations Review & Bridge to Deep Learning"
description: "Review ML Foundations and see why linear models fail on complex patterns — motivation for neural networks."
date: 2026-03-20T00:00:00Z
weight: 100
draft: false
difficulty: 4
tags: ["review", "bridge", "neural networks", "deep learning", "ml-foundations"]
keywords: ["ML review", "bridge to deep learning", "XOR problem", "non-linear boundaries", "neural networks motivation"]
roadmap_icon: "globe"
roadmap_color: "violet"
roadmap_phase_label: "Review & Bridge"
---

**Learning objectives**

- Recall key ML Foundations concepts and their RL connections.
- Demonstrate that logistic regression cannot solve XOR (non-linearly separable data).
- Articulate why neural networks are the natural next step after linear models.

## ML Foundations Recap Quiz

Five questions. Attempt each before revealing the answer.

**Q1.** What is the difference between classification and regression?

{{< collapse summary="Q1 answer" >}}
**Regression** predicts a continuous value (e.g. house price, Q-value in RL). **Classification** predicts a discrete class label (e.g. spam/not-spam, or which action to take from a finite action set). Logistic regression is a classifier despite its name — it outputs a probability that gets thresholded.
{{< /collapse >}}

**Q2.** What does gradient descent minimize? Write the update rule.

{{< collapse summary="Q2 answer" >}}
Gradient descent minimizes the **loss function** \\(\mathcal{L}(w)\\) (e.g. MSE for regression, cross-entropy for classification). Update rule:
\\[w \leftarrow w - \alpha \frac{\partial \mathcal{L}}{\partial w}\\]
where \\(\alpha\\) is the learning rate. Each step moves parameters in the direction that most reduces the loss.
{{< /collapse >}}

**Q3.** When does a linear model fail?

{{< collapse summary="Q3 answer" >}}
A linear model fails when the true decision boundary (for classification) or the true mapping (for regression) is **non-linear**. For example: XOR — no single straight line separates the two classes. Concentric circles, spiral datasets, and any problem where the output depends on interactions between features that a linear combination cannot capture. In RL, value functions over raw pixel observations are non-linear — a linear model cannot represent them without hand-crafted features.
{{< /collapse >}}

**Q4.** Why do we split data into train and test sets?

{{< collapse summary="Q4 answer" >}}
To get an **honest estimate of generalization performance**. A model trained and evaluated on the same data can memorize labels without learning patterns. The test set, seen only at evaluation time, reveals whether the model generalizes. In RL terms: we evaluate agents on new seeds/environments they did not train on to measure true policy quality.
{{< /collapse >}}

**Q5.** In 3 sentences, explain what cross-validation is.

{{< collapse summary="Q5 answer" >}}
Cross-validation splits data into K equal folds. For each fold, you train on the remaining K-1 folds and evaluate on the held-out fold. The K test scores are averaged to give a stable, low-variance estimate of model quality — more reliable than any single train/test split.
{{< /collapse >}}

---

## What Changes in Deep Learning

Linear models are powerful — but limited. Here is what changes when we move to neural networks.

| | Linear / Logistic Regression | Neural Networks |
|---|---|---|
| Can model non-linear boundaries? | No | Yes |
| Number of parameters | Small (one per feature) | Large (millions possible) |
| Training method | Gradient descent | Gradient descent + backpropagation |
| Interpretability | High (inspect weights directly) | Low (black box) |
| Good for RL value functions? | Only with hand-crafted features | Yes — can use raw states |
| Risk of overfitting | Low | High (needs regularization) |

The key insight: **the training algorithm is the same.** Neural networks use gradient descent too — but the gradient flows through multiple layers via **backpropagation** (the chain rule applied recursively). Everything you learned about loss functions, learning rates, overfitting, and evaluation applies directly.

---

## Bridge Exercise

**The XOR problem:** Logistic regression fails on data that is not linearly separable. XOR is the canonical example — no straight line can separate the two classes.

{{< pyrepl code="from sklearn.linear_model import LogisticRegression\nimport numpy as np\n\n# XOR data: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0\nX_xor = np.array([[0,0],[0,1],[1,0],[1,1]])\ny_xor = np.array([0, 1, 1, 0])\n\n# Train logistic regression\nlr = LogisticRegression()\nlr.fit(X_xor, y_xor)\n\ny_pred = lr.predict(X_xor)\nacc = np.mean(y_pred == y_xor)\n\nprint('XOR predictions:', y_pred)\nprint('True labels:     ', y_xor)\nprint(f'Accuracy: {acc:.2f}')\nprint('\\nConclusion:')\nif acc <= 0.5:\n    print('Logistic regression achieves only 50% — no better than random.')\n    print('This is why we need neural networks.')\nelse:\n    print(f'Got {acc:.0%} — but try different random states.')\n\n# This is why we need neural networks — they can learn non-linear boundaries.\n# A 2-layer net with a hidden layer of 2 neurons solves XOR perfectly." height="280" >}}

{{< collapse summary="Why XOR is unsolvable by logistic regression" >}}
Logistic regression draws a single linear decision boundary (a line in 2D). For XOR, the four points `(0,0)→0`, `(0,1)→1`, `(1,0)→1`, `(1,1)→0` cannot be separated by any line — the "1" class and "0" class alternate in a checkerboard pattern. Logistic regression is stuck at 50% accuracy regardless of training.

A **neural network with one hidden layer** solves XOR by learning a non-linear feature transformation first, then classifying in the transformed space. This is the core motivation for deep learning.

```python
# A network that solves XOR:
# Hidden layer: h1 = relu(w1 @ x + b1)  ← non-linear transformation
# Output:       y  = sigmoid(w2 @ h1 + b2)
# The hidden layer creates a new feature space where XOR IS linearly separable.
```
{{< /collapse >}}

---

## Ready for Deep Learning?

Check off each item honestly before moving on:

- [ ] I can implement MSE, compute its gradient, and take one gradient descent step.
- [ ] I understand train/test split and why evaluating on training data is wrong.
- [ ] I know what sigmoid does and when to use cross-entropy loss.
- [ ] I built a full sklearn pipeline and compared multiple models.
- [ ] I understand why linear models are limited — and what XOR illustrates.
- [ ] I can explain K-fold cross-validation and the bias-variance tradeoff.
- [ ] I implemented KNN and K-Means from scratch in NumPy.

**If you checked all 7:** You are ready.

**If you missed any:** Revisit the relevant page before continuing. The next section (Deep Learning Foundations) builds directly on all of these.

**Next:** [DL Foundations](../../dl-foundations/)
