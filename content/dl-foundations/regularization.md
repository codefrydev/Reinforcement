---
title: "Regularization and Overfitting"
description: "Understand overfitting and apply L2 regularization and dropout to prevent it in NumPy."
date: 2026-03-20T00:00:00Z
weight: 10
draft: false
difficulty: 5
tags: ["deep learning", "regularization", "overfitting", "dropout", "L2", "dl-foundations"]
keywords: ["overfitting", "L2 regularization", "dropout", "early stopping", "neural network generalization"]
roadmap_icon: "database"
roadmap_color: "teal"
roadmap_phase_label: "Chapter 10"
---

**Learning objectives**
- Recognize overfitting from learning curves and understand why it happens
- Implement L2 regularization: add the penalty to the loss and adjust the gradient
- Implement dropout: randomly zero out neurons during training
- Understand when each technique is appropriate and how they connect to RL

**Concept and real-world motivation**

**Overfitting** happens when a model memorizes the training data instead of learning generalizable patterns. The training loss keeps decreasing, but the validation loss starts increasing — the model has "overfit." This is especially easy to trigger with large networks on small datasets.

The main fixes: **L2 regularization** penalizes large weights, encouraging the model to use small distributed representations. **Dropout** randomly disables neurons during training, preventing co-adaptation — neurons can't rely on each other and must learn independently useful features. **Early stopping** halts training when validation loss stops improving.

**In RL:** Overfitting in RL is called **overspecialization** — the agent memorizes specific environment states or transitions instead of generalizing. DQN uses target networks and replay buffers partly to reduce this. Policy networks in PPO often use entropy bonuses to avoid overconfident (overfit) policies.

**Math:**

L2 regularization: \\(L_{reg} = L + \frac{\lambda}{2}\|w\|^2\\)

Gradient with L2: \\(\nabla L_{reg} = \nabla L + \lambda w\\)

Dropout during training: randomly zero out each neuron with probability \\(p\\), then scale the surviving activations by \\(\frac{1}{1-p}\\) (inverted dropout) so expected values are unchanged at test time.

**Illustration — Overfitting: train loss vs validation loss:**

{{< chart type="line" palette="learning" title="Train loss (decreasing) vs Validation loss (increasing after epoch 20)" labels="5,10,15,20,25,30" data="0.8,0.6,0.4,0.3,0.25,0.2" xLabel="Epoch" yLabel="Loss (train)" >}}

The validation loss follows the train loss early, then diverges — this is the overfitting signal. Stop training at the validation loss minimum.

**Exercise:** Add L2 regularization to a 1-layer training loop and compare final weights.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n\n# Simple regression: y = 2x + noise\nX = np.random.randn(50, 3)  # 50 samples, 3 features\ntrue_w = np.array([2.0, 0.0, 0.0])  # only first feature matters\ny = X @ true_w + 0.1 * np.random.randn(50)\n\ndef train(lam, epochs=200, lr=0.01):\n    w = np.zeros(3)\n    for _ in range(epochs):\n        pred = X @ w\n        loss = np.mean((pred - y) ** 2) + (lam / 2) * np.sum(w ** 2)\n        grad = 2 * X.T @ (pred - y) / len(y) + lam * w\n        w -= lr * grad\n    return w\n\nw_no_reg = train(lam=0.0)\nw_reg = train(lam=0.1)\n\nprint('No regularization:', [round(x, 4) for x in w_no_reg])\nprint('L2 reg (lam=0.1): ', [round(x, 4) for x in w_reg])\nprint('Note: regularized weights are smaller in magnitude')" height="280" >}}

**Professor's hints**
- The regularization strength \\(\lambda\\) is a hyperparameter: too large shrinks everything to zero; too small has no effect. Typical values: 1e-4 to 1e-2.
- Dropout rate \\(p=0.5\\) is common for fully-connected layers; \\(p=0.1\\) to \\(0.3\\) for convolutional layers.
- Always turn off dropout at test/evaluation time — only apply it during training.
- L2 regularization is equivalent to placing a Gaussian prior over weights (MAP estimation).

**Common pitfalls**
- Applying dropout during evaluation — this is a common bug that degrades test performance unpredictably.
- Forgetting to scale activations after dropout (inverted dropout ensures the same expected activation magnitude).
- Using L2 on bias terms — convention is to regularize weights but not biases.

{{< collapse summary="Worked solution" >}}
L2 adds `lambda * w` to the gradient. This "weight decay" pulls weights toward zero during every update, preventing any weight from growing very large.

For dropout with inverted scaling:
```python
def dropout(a, p=0.5, training=True):
    if not training:
        return a  # no dropout at test time
    mask = (np.random.rand(*a.shape) > p).astype(float)
    return (a * mask) / (1 - p)  # scale to preserve expected value
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Implement dropout mask in NumPy — randomly set neurons to 0 with p=0.5, then apply inverted scaling.
{{< pyrepl code="import numpy as np\nnp.random.seed(0)\na = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # neuron activations\np = 0.5  # dropout probability\n# TODO: create mask, apply dropout, scale by 1/(1-p)\n# expected: roughly half the values are zero, rest are doubled\nmask = (np.random.rand(len(a)) > p).astype(float)\na_dropped = (a * mask) / (1 - p)\nprint('mask:', mask)\nprint('dropped activations:', a_dropped)\nprint('original mean:', a.mean(), '  dropped mean (approx):', a_dropped.mean())" height="200" >}}

2. **Coding:** Add validation set tracking to the regularization training loop. Plot (conceptually) train loss and val loss. At which epoch does the validation loss stop decreasing?

3. **Challenge:** Implement early stopping: monitor validation loss and stop training if it hasn't improved for 10 consecutive epochs. Return the weights from the best epoch.

4. **Variant:** Compare L1 regularization (\\(\lambda |w|\\)) vs L2 (\\(\lambda w^2/2\\)) on sparse data. L1 encourages exact zeros; L2 shrinks weights smoothly. Implement both and compare weight histograms.

5. **Debug:** Fix the dropout below that is applied during evaluation:
{{< pyrepl code="import numpy as np\nnp.random.seed(1)\n\ndef forward(X, W, b, training=True):\n    z = X @ W + b\n    a = np.maximum(0, z)\n    # BUG: dropout applied regardless of training flag\n    mask = (np.random.rand(*a.shape) > 0.5)\n    a = a * mask / 0.5\n    return a\n\nX_test = np.ones((5, 3))\nW = np.ones((3, 2))\nb = np.zeros(2)\n# Evaluation should be deterministic\nout1 = forward(X_test, W, b, training=False)\nout2 = forward(X_test, W, b, training=False)\nprint('Outputs equal?', np.allclose(out1, out2), '  (should be True for eval)')\n# TODO: fix forward to only apply dropout when training=True" height="220" >}}

6. **Conceptual:** In RL, the policy network is evaluated online during environment interaction. Why is it critical that dropout is disabled (inference mode) during rollout? What would happen if it weren't?

7. **Recall:** Name three regularization techniques and describe in one sentence how each prevents overfitting.
