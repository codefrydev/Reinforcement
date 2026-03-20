---
title: "Logistic Regression"
description: "Binary classifier from scratch: sigmoid + cross-entropy loss + gradient update. The building block of softmax policies."
date: 2026-03-20T00:00:00Z
weight: 7
draft: false
difficulty: 4
tags: ["logistic regression", "cross-entropy", "classification", "gradient descent", "ml-foundations"]
keywords: ["logistic regression from scratch", "cross-entropy loss", "binary classification gradient", "softmax policy", "NumPy logistic regression", "RL policy gradient"]
roadmap_icon: "calculator"
roadmap_color: "indigo"
roadmap_phase_label: "Chapter 7"
---

**Learning objectives**

- Derive the cross-entropy loss for binary classification and explain why it is preferred over MSE for classifiers.
- Compute the gradient of cross-entropy with respect to \\(w\\) in matrix form.
- Implement logistic regression training from scratch in NumPy and observe the loss decreasing over iterations.

**Concept and real-world motivation**

Logistic regression combines three components you already know: (1) the **linear model** \\(z = Xw + b\\), (2) the **sigmoid** \\(p = \sigma(z)\\), and (3) a new loss function designed for probabilities. Using MSE for classification would make the loss surface very flat near 0 and 1, making gradients vanishingly small. The right loss for probabilities is **cross-entropy**:

\\[L = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i) \right]\\]

This loss is large when the model is confident and wrong, and near zero when the model is correct. The elegant fact about logistic regression is that its gradient has a beautifully simple form:

\\[\nabla_w L = \frac{1}{n} X^T (\hat{p} - y)\\]

This is the same structure as linear regression's gradient — the only difference is the "residual" is \\(\hat{p} - y\\) instead of \\(\hat{y} - y\\).

**RL connection:** The **softmax policy** in RL is logistic regression generalised to multiple actions. The policy network computes \\(z = W^T s\\) (one value per action), passes it through softmax, and outputs action probabilities \\(\pi(a \mid s)\\). The REINFORCE policy gradient objective is the expected log probability of chosen actions — cross-entropy in disguise. Mastering logistic regression here means policy gradient is just the same math with a different dataset.

**Illustration:** The cross-entropy loss drops as logistic regression training progresses.

{{< chart type="line" palette="learning" title="Cross-entropy loss during logistic regression training" labels="0, 10, 20, 30, 40, 50" data="0.693, 0.521, 0.413, 0.341, 0.291, 0.254" xLabel="Training step" yLabel="Cross-entropy loss" >}}

Before implementing training, verify the loss formula by hand:

{{< pyrepl code="import numpy as np\n\n# Compute cross-entropy by hand for 3 predictions\ny_true = np.array([1, 0, 1])          # ground truth labels\ny_pred = np.array([0.8, 0.3, 0.6])   # model probabilities\n\n# cross-entropy = -mean(y*log(p) + (1-y)*log(1-p))\nloss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))\nprint(f'Cross-entropy loss: {loss:.4f}')\n# expected: ~0.3567 (lower than log(2)=0.693 because predictions are good)\n\n# Compare: what if predictions were terrible?\ny_pred_bad = np.array([0.1, 0.9, 0.1])\nloss_bad = -np.mean(y_true * np.log(y_pred_bad) + (1 - y_true) * np.log(1 - y_pred_bad))\nprint(f'Bad predictions loss: {loss_bad:.4f}')  # expected: much larger" height="220" >}}

**Exercise:** Implement full logistic regression training on a toy dataset. Forward pass, cross-entropy loss, gradient computation, and weight update — all in NumPy.

{{< pyrepl code="import numpy as np\n\nnp.random.seed(42)\n\n# Toy dataset: 3 features, 5 samples, binary labels\nX = np.array([\n    [1.0, 2.0, 0.5],\n    [2.0, 1.0, 1.5],\n    [0.5, 3.0, 0.2],\n    [3.0, 0.5, 2.0],\n    [1.5, 1.5, 1.0],\n])\ny = np.array([1, 1, 0, 1, 0], dtype=float)\n\n# TODO: implement sigmoid\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\n# Initialise weights\nw = np.zeros(3)\nb = 0.0\nlr = 0.1\nn  = len(y)\n\n# TODO: run 100 training steps\nfor step in range(100):\n    # TODO: forward pass: z = X @ w + b, p = sigmoid(z)\n    z = None\n    p = None\n\n    # TODO: cross-entropy loss\n    loss = None\n\n    # TODO: gradient: grad_w = (1/n) * X.T @ (p - y)\n    grad_w = None\n    grad_b = None\n\n    # TODO: update w and b (SUBTRACT the gradient to minimise loss)\n    w = None\n    b = None\n\n    if step % 20 == 0:\n        print(f'step {step:3d}: loss={loss:.4f}' if loss is not None else f'step {step}: loss=None')\n\n# expected: loss should decrease from ~0.69 to around 0.20 by step 80" height="360" >}}

**Professor's hints**

- Forward: `z = X @ w + b` (shape `(n,)`), then `p = sigmoid(z)` (shape `(n,)`).
- Cross-entropy: `loss = -np.mean(y * np.log(p) + (1-y) * np.log(1-p))` — add a small epsilon (e.g. `1e-9`) inside `log` if you hit numerical warnings.
- Gradient w.r.t. w: `(1/n) * X.T @ (p - y)` — shape `(d,)`.
- Gradient w.r.t. b: `(1/n) * np.sum(p - y)` — scalar.
- Update: `w = w - lr * grad_w`. Note: subtract (we minimise loss), not add.

**Common pitfalls**

- **Adding the gradient instead of subtracting:** `w = w + lr * grad_w` is gradient *ascent* — it maximises the loss, which is the wrong direction for a classifier.
- **Taking log of zero:** When the model is very confident and wrong (\\(\hat{p} \approx 0\\) but \\(y=1\\)), \\(\log(0) = -\infty\\). Add a small epsilon: `np.log(p + 1e-9)`.
- **Forgetting to include the \\((1-y)\log(1-\hat{p})\\) term:** The cross-entropy formula has two terms — one for positive examples (\\(y=1\\)) and one for negative (\\(y=0\\)). Missing either term silently breaks the loss.

{{< collapse summary="Worked solution" >}}
Complete logistic regression from scratch:

```python
import numpy as np

np.random.seed(42)
X = np.array([[1.,2.,0.5],[2.,1.,1.5],[0.5,3.,0.2],[3.,0.5,2.],[1.5,1.5,1.]])
y = np.array([1,1,0,1,0], dtype=float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w  = np.zeros(3)
b  = 0.0
lr = 0.1
n  = len(y)

for step in range(100):
    # Forward
    z    = X @ w + b
    p    = sigmoid(z)
    # Loss
    loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
    # Gradients
    grad_w = (1/n) * X.T @ (p - y)
    grad_b = (1/n) * np.sum(p - y)
    # Update
    w = w - lr * grad_w
    b = b - lr * grad_b
    if step % 20 == 0:
        print(f'step {step:3d}: loss={loss:.4f}')
```

Expected output: loss decreases from ~0.6931 at step 0 to ~0.20 by step 80.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For labels `y=[1, 0]` and predicted probabilities `p=[0.9, 0.1]`, compute the cross-entropy by hand. Then try `p=[0.5, 0.5]`. Which has higher loss and why?
2. **Coding:** Use `sklearn.linear_model.LogisticRegression` on the same toy dataset. Compare `model.coef_` and `model.intercept_` to the weights your NumPy implementation converges to after 500 steps.
3. **Challenge:** Extend the implementation to multi-class classification using softmax instead of sigmoid. For 3 classes, replace \\(w \in \mathbb{R}^d\\) with \\(W \in \mathbb{R}^{d \times 3}\\) and compute \\(\text{softmax}(Xw)\\). Use cross-entropy for 3 classes.
4. **Variant:** Try learning rates `lr = 0.01, 0.1, 1.0`. Plot the loss curve for each. What happens with `lr=1.0` — does it converge, oscillate, or diverge?
5. **Debug:** The gradient update below *adds* the gradient instead of subtracting it, causing the loss to increase instead of decrease. Fix it.

{{< pyrepl code="import numpy as np\n\ndef sigmoid(z): return 1 / (1 + np.exp(-z))\n\nX = np.array([[1.,2.],[2.,1.],[0.5,3.],[3.,0.5]])\ny = np.array([1., 1., 0., 1.])\nw = np.zeros(2)\nb = 0.0\nlr = 0.1\nn = len(y)\n\nfor step in range(10):\n    z    = X @ w + b\n    p    = sigmoid(z)\n    loss = -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))\n    grad_w = (1/n) * X.T @ (p - y)\n    grad_b = (1/n) * np.sum(p - y)\n    w = w + lr * grad_w   # BUG: should subtract\n    b = b + lr * grad_b   # BUG: should subtract\n    print(f'step {step}: loss={loss:.4f}')\n\n# expected: loss should decrease\n# with bug:  loss increases toward infinity\n# TODO: fix both update lines" height="240" >}}

6. **Conceptual:** Logistic regression is a linear model — the decision boundary is always a straight line (or hyperplane). When would this be a problem? Name a dataset where logistic regression would fail and a model that could succeed.
7. **Recall:** Write the cross-entropy loss formula for binary classification from memory. Then write the gradient \\(\nabla_w L\\) in matrix form.
