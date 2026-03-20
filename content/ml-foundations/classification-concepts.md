---
title: "Classification Concepts"
description: "Predict categories instead of numbers. Decision boundaries, sigmoid activation, and binary probability outputs."
date: 2026-03-20T00:00:00Z
weight: 6
draft: false
difficulty: 4
tags: ["classification", "decision boundary", "sigmoid", "binary classification", "ml-foundations"]
keywords: ["binary classification", "sigmoid function", "decision boundary", "classification vs regression", "probability output", "RL policy"]
roadmap_icon: "network"
roadmap_color: "rose"
roadmap_phase_label: "Chapter 6"
---

**Learning objectives**

- Distinguish regression (continuous output) from classification (categorical output).
- Compute the sigmoid function \\(\sigma(z) = \frac{1}{1+e^{-z}}\\) and interpret its output as a probability.
- Apply a decision threshold (\\(p > 0.5\\)) to convert probabilities into class predictions.

**Concept and real-world motivation**

Regression predicts a *number* (price, temperature, distance). **Classification** predicts a *category* (spam/not-spam, dog/cat, buy/sell). The key challenge in classification is converting the linear output \\(z = w^T x + b\\) — which can be any real number — into a probability between 0 and 1.

The **sigmoid function** does exactly this:

\\[\sigma(z) = \frac{1}{1 + e^{-z}}\\]

When \\(z\\) is very large and positive, \\(\sigma(z) \approx 1\\) (certain positive). When \\(z\\) is very large and negative, \\(\sigma(z) \approx 0\\) (certain negative). At \\(z=0\\), \\(\sigma(0) = 0.5\\) (maximum uncertainty). The **decision boundary** is the set of points where \\(\sigma(z) = 0.5\\), which is where \\(z = w^T x + b = 0\\).

In binary classification, the standard rule is:
- If \\(\sigma(z) > 0.5\\) → predict class 1 (positive)
- If \\(\sigma(z) \leq 0.5\\) → predict class 0 (negative)

**RL connection:** The policy \\(\pi(a \mid s)\\) in RL is a *probability distribution over actions* — it is a classifier. Given the state \\(s\\), the policy outputs probabilities for each action: \\(\pi(\text{left} \mid s) = 0.3, \pi(\text{right} \mid s) = 0.7\\). The softmax function (a multi-class generalisation of sigmoid) converts a linear output \\(z = w^T s\\) into action probabilities. Sigmoid and softmax are the same idea — sigmoid for binary, softmax for multi-class.

**Illustration:** The sigmoid function maps any real number \\(z\\) to a value between 0 and 1.

{{< chart type="bar" palette="math" title="Sigmoid output σ(z) for different z values" labels="-3, -2, -1, 0, 1, 2, 3" data="0.05, 0.12, 0.27, 0.50, 0.73, 0.88, 0.95" xLabel="z (linear output)" yLabel="σ(z) = probability" >}}

**Exercise:** Implement sigmoid in NumPy. Given 6 data points with z-scores, compute their probabilities and classify each as positive (1) or negative (0) using threshold \\(p > 0.5\\).

{{< pyrepl code="import numpy as np\n\n# TODO: implement the sigmoid function\ndef sigmoid(z):\n    # sigma(z) = 1 / (1 + exp(-z))\n    pass\n\n# 6 test inputs\nz_scores = np.array([-3.0, -1.0, -0.5, 0.5, 1.0, 3.0])\n\n# TODO: compute probabilities using sigmoid\nprobs = None\n\n# TODO: classify: 1 if prob > 0.5, else 0\npredictions = None\n\nprint('z_scores:   ', z_scores)\nprint('probs:      ', probs.round(3) if probs is not None else None)\nprint('predictions:', predictions)\n# expected probs:   [0.047, 0.269, 0.378, 0.622, 0.731, 0.953]\n# expected classes: [0, 0, 0, 1, 1, 1]" height="260" >}}

**Professor's hints**

- `sigmoid(z) = 1 / (1 + np.exp(-z))` — use `np.exp` for NumPy arrays (not `math.exp`).
- `(probs > 0.5).astype(int)` converts a boolean array to 0/1 integers — clean one-liner for threshold classification.
- `probs.round(3)` rounds to 3 decimal places for readable output.
- Notice that z=0 gives prob=0.5 exactly — the model is maximally uncertain at the decision boundary.

**Common pitfalls**

- **Using `math.exp` instead of `np.exp`:** `math.exp` only works on a single scalar. For NumPy arrays, always use `np.exp(-z)`.
- **Confusing \\(z\\) with \\(p\\):** \\(z = w^T x + b\\) is the raw linear output (can be any real number). \\(p = \sigma(z)\\) is the probability (always between 0 and 1). They are different quantities.
- **Threshold of 0.5 is not always optimal:** In class-imbalanced problems (e.g. 1% positive rate), using 0.5 as the threshold often misses positive examples. The threshold should be chosen based on the cost of false positives vs false negatives.

{{< collapse summary="Worked solution" >}}
Sigmoid implementation and classification:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_scores = np.array([-3.0, -1.0, -0.5, 0.5, 1.0, 3.0])

probs       = sigmoid(z_scores)
# [0.047, 0.269, 0.378, 0.622, 0.731, 0.953]

predictions = (probs > 0.5).astype(int)
# [0, 0, 0, 1, 1, 1]
```

The decision boundary is at z=0 (prob=0.5). Any sample with a positive \\(z\\)-score is classified as 1.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Compute \\(\sigma(0)\\), \\(\sigma(1)\\), and \\(\sigma(-1)\\) by hand (substitute into the formula). Verify your answers using the bar chart above. Then write a 3-line Python snippet to confirm.

{{< pyrepl code="import numpy as np\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\n# TODO: compute and print sigmoid(0), sigmoid(1), sigmoid(-1)\n# expected: 0.5, 0.731, 0.269\nprint(sigmoid(0))\nprint(sigmoid(1))\nprint(sigmoid(-1))" height="160" >}}

2. **Coding:** Plot the sigmoid function for \\(z \in [-6, 6]\\) using 100 evenly spaced points. Use matplotlib (`plt.plot`). Add a horizontal dashed line at y=0.5 to show the decision boundary.
3. **Challenge:** The sigmoid of very large values can cause numerical issues (`exp(-1000)` underflows to 0 on some systems, which is fine; but `exp(1000)` overflows). Implement a numerically stable version of sigmoid that uses `np.clip(z, -500, 500)` before computing. Verify it handles z=1000 and z=-1000 correctly.
4. **Variant:** Change the threshold from 0.5 to 0.7. How does this change the predictions on the 6 z-scores in the exercise? When would you want a higher threshold?
5. **Debug:** The code below uses the wrong threshold — it classifies as positive if prob < 0.5 instead of prob > 0.5. Fix it.

{{< pyrepl code="import numpy as np\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\nz_scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])\nprobs = sigmoid(z_scores)\n\n# BUG: threshold is inverted\npredictions = (probs < 0.5).astype(int)   # BUG: should be >\n\nprint('probs:      ', probs.round(3))\nprint('predictions:', predictions)\n# expected predictions: [0, 0, 0, 1, 1]\n# actual (with bug):    [1, 1, 1, 0, 0]\n# TODO: fix the threshold comparison" height="200" >}}

6. **Conceptual:** Why can we not use MSE as the loss function for classification? Sketch why the loss surface becomes flat and the gradient vanishes for a classifier trained with MSE + sigmoid.
7. **Recall:** Write the sigmoid formula from memory. What is \\(\sigma(0)\\)? What happens as \\(z \to +\infty\\) and \\(z \to -\infty\\)?
