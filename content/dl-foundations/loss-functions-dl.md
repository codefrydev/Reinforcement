---
title: "Loss Functions: Measuring How Wrong the Network Is"
description: "MSE for regression, cross-entropy for classification, and the TD error loss in DQN — how loss functions guide neural network training."
date: 2026-03-20T00:00:00Z
weight: 6
draft: false
difficulty: 5
tags: ["loss functions", "MSE", "cross-entropy", "training", "dl-foundations"]
keywords: ["MSE loss", "cross-entropy loss", "binary cross-entropy", "loss function neural network", "DQN loss TD error"]
roadmap_icon: "chart"
roadmap_color: "rose"
roadmap_phase_label: "Chapter 6"
---

**Learning objectives**

- Implement mean squared error (MSE) and binary cross-entropy loss in NumPy.
- Explain when to use each loss function and connect them to their RL equivalents.
- Identify the numerical stability issue in cross-entropy and fix it with an epsilon clamp.

**Concept and real-world motivation**

A neural network learns by minimizing a **loss function** — a scalar that measures how wrong the current predictions are. The loss function is the signal that backpropagation differentiates to compute gradients. Choose the wrong loss for your task, and the network will optimize for the wrong thing.

**Mean Squared Error (MSE)** is the standard loss for regression tasks — when the output is a continuous number:
\\[L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2\\]

**Cross-Entropy (CE)** is the standard loss for classification tasks — when the output is a probability distribution:
\\[L_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c} y_{ic} \log(\hat{p}_{ic})\\]

**Binary Cross-Entropy (BCE)** is a special case for binary (two-class) problems:
\\[L_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]\\]

**In RL:**
- **DQN uses MSE** between the predicted Q-value and the Bellman target: \\(L = (r + \gamma \max_{a'} Q(s', a') - Q(s, a))^2\\). This is regression — predicting a scalar value.
- **Policy gradient methods** use a variant of cross-entropy: the policy gradient loss maximizes the log probability of good actions: \\(L = -\log \pi(a|s) \cdot G_t\\).

The choice of loss is not arbitrary — it must match the output distribution. Q-values are unbounded real numbers → MSE. Action probabilities are distributions over discrete actions → cross-entropy.

**Illustration:**

{{< chart type="bar" palette="comparison" title="Loss values at start vs end of training" labels="MSE start, CE start, MSE end, CE end" data="0.45, 0.68, 0.02, 0.08" xLabel="Loss metric" yLabel="Value" >}}

**Exercise:** Implement MSE and binary cross-entropy in NumPy. Verify that cross-entropy is lower when predictions are more confident and correct.

{{< pyrepl code="import numpy as np\n\n# --- Binary cross-entropy ---\ntrue_labels  = np.array([1, 0, 1, 1, 0])\npredictions  = np.array([0.9, 0.2, 0.7, 0.8, 0.3])  # predicted probabilities\n\n# TODO: implement binary cross-entropy\n# L = -mean(y * log(p) + (1-y) * log(1-p))\ndef binary_cross_entropy(y_true, y_pred):\n    pass  # hint: use np.log; be careful of log(0)\n\nbce = binary_cross_entropy(true_labels, predictions)\nprint(f'Binary cross-entropy: {bce:.4f}')\n\n# --- Mean Squared Error ---\nreg_true = np.array([2.5, 3.0, 1.5, 4.0])\nreg_pred = np.array([2.3, 3.2, 1.8, 3.7])\n\n# TODO: implement MSE\ndef mse(y_true, y_pred):\n    pass  # hint: np.mean((y_true - y_pred)**2)\n\nmse_val = mse(reg_true, reg_pred)\nprint(f'MSE: {mse_val:.4f}')\n\n# Verify: confident correct predictions → lower CE\nbad_preds = np.array([0.6, 0.5, 0.55, 0.6, 0.45])\nbce_bad = binary_cross_entropy(true_labels, bad_preds)\nprint(f'BCE (confident preds): {bce:.4f}')\nprint(f'BCE (uncertain preds): {bce_bad:.4f}')\nprint(f'Confident < Uncertain: {bce < bce_bad}')\n# expected: True — lower BCE for better predictions" height="260" >}}

**Professor's hints**

- MSE: `np.mean((y_true - y_pred)**2)`. Simple.
- Binary CE: `np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))`.
- If `y_pred` is exactly 0 or 1, `np.log(0)` = -inf. Always clip: `np.clip(y_pred, 1e-7, 1 - 1e-7)`.
- Cross-entropy is always ≥ 0. MSE can theoretically be 0 if predictions are perfect.

**Common pitfalls**

- **Taking log of 0:** `np.log(0) = -inf` and `-inf * 0 = nan`. Always add a small epsilon or use `np.clip`.
- **Using MSE for classification:** MSE doesn't work well for classification because the probability outputs don't interact with the loss correctly — use cross-entropy instead.
- **Forgetting the minus sign in CE:** Cross-entropy has a leading negative sign because \\(\log p < 0\\) for \\(p \in (0,1)\\). Without it, you'd be maximizing the loss.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np

true_labels = np.array([1, 0, 1, 1, 0])
predictions = np.array([0.9, 0.2, 0.7, 0.8, 0.3])

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # numerical stability
    return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

print(f'BCE: {binary_cross_entropy(true_labels, predictions):.4f}')  # ≈ 0.1879
print(f'MSE: {mse(np.array([2.5,3.0,1.5,4.0]), np.array([2.3,3.2,1.8,3.7])):.4f}')  # ≈ 0.055
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Compute MSE by hand for predictions `[2.0, 4.0]` vs true values `[3.0, 3.0]`. Then verify with NumPy.

{{< pyrepl code="import numpy as np\n\ny_true = np.array([3.0, 3.0])\ny_pred = np.array([2.0, 4.0])\n\n# By hand: MSE = ((3-2)^2 + (3-4)^2) / 2 = (1 + 1) / 2 = 1.0\nmse_manual = None  # TODO: compute using np.mean\n\nprint(f'MSE: {mse_manual:.4f}')\n# expected: 1.0000" height="150" >}}

2. **Coding:** Implement multi-class cross-entropy for a 3-class problem. Given `y_true = [[1,0,0],[0,1,0],[0,0,1]]` (one-hot) and `y_pred = [[0.7,0.2,0.1],[0.1,0.8,0.1],[0.2,0.2,0.6]]`, compute the loss.
3. **Challenge:** The DQN loss is \\((r + \gamma \max_{a'} Q(s',a') - Q(s,a))^2\\). This looks like MSE with `y_true = r + γ * max Q(s',a')` and `y_pred = Q(s,a)`. Implement this "TD error" loss for a batch of 4 transitions with random Q-values.
4. **Variant:** Huber loss combines MSE and MAE: it is MSE for small errors and MAE for large errors, making it more robust to outliers. Implement it and compare to MSE on predictions `[5.0, 0.1]` vs true `[0.0, 0.0]`.

{{< pyrepl code="import numpy as np\n\ny_true = np.array([0.0, 0.0])\ny_pred = np.array([5.0, 0.1])  # one large error, one small\n\ndef huber_loss(y_true, y_pred, delta=1.0):\n    error = y_pred - y_true\n    # TODO: |error| <= delta → 0.5 * error^2; else → delta*(|error| - 0.5*delta)\n    return np.mean(np.where(np.abs(error) <= delta,\n                            0.5 * error**2,\n                            delta * (np.abs(error) - 0.5 * delta)))\n\ndef mse(y_true, y_pred):\n    return np.mean((y_true - y_pred)**2)\n\nprint(f'MSE:   {mse(y_true, y_pred):.4f}')    # heavily penalizes large error\nprint(f'Huber: {huber_loss(y_true, y_pred):.4f}')  # more robust" height="220" >}}

5. **Debug:** The cross-entropy below will produce `nan` because it takes the log of 0. Fix it by clipping predictions to a small epsilon.

{{< pyrepl code="import numpy as np\n\ny_true = np.array([1, 0, 1, 0])\ny_pred = np.array([1.0, 0.0, 0.8, 0.2])  # contains 0.0 and 1.0\n\n# BUG: no clipping, log(0) = -inf, -inf * 0 = nan\ndef binary_cross_entropy_buggy(y_true, y_pred):\n    # TODO: add np.clip before taking log\n    return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))\n\nloss = binary_cross_entropy_buggy(y_true, y_pred)\nprint(f'Loss: {loss}')\n# expected after fix: a finite positive number (≈ 0.1116)" height="180" >}}

6. **Conceptual:** Why does cross-entropy work better than MSE for classification? Consider what happens to the gradient of MSE when the sigmoid output is near 0 or 1 — the gradient becomes very small. Cross-entropy with sigmoid has a much cleaner gradient. Explain this in one paragraph.
7. **Recall:** Write the MSE and binary cross-entropy formulas from memory. State the task type (regression or classification) each is used for. Name the RL algorithm that uses each.
