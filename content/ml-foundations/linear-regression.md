---
title: "Linear Regression"
description: "Predict a continuous value with ŷ = wx + b. Derive the MSE loss and compute one gradient descent step from scratch."
date: 2026-03-20T00:00:00Z
weight: 3
draft: false
difficulty: 4
tags: ["linear regression", "MSE", "gradient descent", "supervised learning", "ml-foundations"]
keywords: ["linear regression from scratch", "mean squared error", "MSE loss", "gradient descent", "NumPy regression", "value function approximation"]
roadmap_icon: "trend-up"
roadmap_color: "green"
roadmap_phase_label: "Chapter 3"
---

**Learning objectives**

- Write the linear model \\(\hat{y} = wx + b\\) and interpret \\(w\\) (slope) and \\(b\\) (intercept).
- Compute the mean squared error (MSE) loss for a set of predictions.
- Derive and compute the gradient of MSE with respect to \\(w\\), and perform one gradient descent update.

**Concept and real-world motivation**

The simplest supervised learning model is a straight line: \\(\hat{y} = wx + b\\). Given a single input feature \\(x\\) (say, house size in square metres), the model predicts an output \\(\hat{y}\\) (house price). The parameter \\(w\\) is the slope — how much the price changes per extra square metre — and \\(b\\) is the intercept — the baseline price when size is zero.

To find the best \\(w\\) and \\(b\\), we need a **loss function** — a number that measures how wrong our predictions are. The standard choice for regression is **mean squared error (MSE)**:

\\[MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2\\]

Large errors are penalised more than small ones (because of the square). To minimise the MSE, we compute its **gradient** with respect to \\(w\\) and take a small step in the opposite direction. The gradient of MSE with respect to \\(w\\) is:

\\[\frac{\partial MSE}{\partial w} = \frac{-2}{n}\sum_{i=1}^{n} x_i(y_i - \hat{y}_i)\\]

And the update rule is \\(w \leftarrow w - \alpha \cdot \frac{\partial MSE}{\partial w}\\), where \\(\alpha\\) is the **learning rate**.

**RL connection:** Linear function approximation is one of the oldest techniques in RL. The value function \\(V(s) = w^T \phi(s)\\) is exactly linear regression on state features \\(\phi(s)\\). The TD update in linear RL is just one gradient step on an MSE-like loss. Everything you learn here reappears in SARSA and TD(0) with function approximation.

**Illustration:** The chart below shows five data points (house sizes) and the predicted line from a well-fitted model.

{{< chart type="line" palette="learning" title="Linear regression: size vs price" labels="Size 1, Size 2, Size 3, Size 4, Size 5" data="2.1, 3.9, 6.1, 7.8, 10.2" xLabel="House size (arbitrary units)" yLabel="Price (×10⁵)" >}}

**Exercise:** You have house size and price data. Compute the MSE for a given \\(w\\) and \\(b\\), compute the gradient of MSE with respect to \\(w\\), and take one gradient step.

{{< pyrepl code="import numpy as np\n\nsizes  = np.array([50, 75, 100, 125, 150], dtype=float)\nprices = np.array([150000, 210000, 280000, 340000, 400000], dtype=float)\n\nw = 2500.0   # slope: price per sq metre\nb = 25000.0  # intercept\n\n# TODO: compute predictions y_hat = w * sizes + b\ny_hat = None\n\n# TODO: compute MSE = mean of (prices - y_hat)^2\nmse = None\n\n# TODO: compute gradient of MSE w.r.t. w\n# grad_w = (-2/n) * sum(sizes * (prices - y_hat))\ngrad_w = None\n\n# TODO: update w with learning rate lr = 0.0001\nlr = 0.0001\nw_new = None\n\nprint(f'MSE:     {mse:.2f}')       # expected: ~0 if w,b are perfect\nprint(f'grad_w:  {grad_w:.4f}')   # expected: small number near 0\nprint(f'w_new:   {w_new:.4f}')    # expected: close to 2500" height="300" >}}

**Professor's hints**

- `y_hat = w * sizes + b` — NumPy broadcasts element-wise automatically.
- MSE: `np.mean((prices - y_hat) ** 2)` — the `**` operator squares element-wise.
- Gradient: `(-2 / len(sizes)) * np.sum(sizes * (prices - y_hat))` — note the element-wise product before summing.
- The learning rate `lr=0.0001` is very small because the features (sizes) are large numbers; large features make large gradients.

**Common pitfalls**

- **Forgetting the \\(-\\) sign in the gradient:** The gradient formula has \\(-(y_i - \hat{y}_i)\\), which equals \\((\hat{y}_i - y_i)\\). Getting the sign wrong makes the model diverge instead of converge.
- **Using too large a learning rate:** If \\(\alpha\\) is too big, one step overshoots the minimum and MSE increases. Always check that MSE goes down after an update.
- **Not normalising features:** When \\(x\\) values are large (like house sizes in square metres), the gradient is also large. Feature normalisation (subtract mean, divide by std) makes training much more stable.

{{< collapse summary="Worked solution" >}}
Step-by-step implementation:

```python
import numpy as np

sizes  = np.array([50, 75, 100, 125, 150], dtype=float)
prices = np.array([150000, 210000, 280000, 340000, 400000], dtype=float)

w = 2500.0
b = 25000.0
lr = 0.0001

# Predictions
y_hat = w * sizes + b
# [150000, 212500, 275000, 337500, 400000]

# MSE
errors = prices - y_hat
mse = np.mean(errors ** 2)
# = mean([0, 6250000, 25000000, 6250000, 0]) = 7500000

# Gradient w.r.t. w
n = len(sizes)
grad_w = (-2 / n) * np.sum(sizes * errors)
# = (-2/5) * (0 + (-187500) + (-2500000) + (-781250) + 0) = 1388500

# One gradient step
w_new = w - lr * grad_w
# = 2500 - 0.0001 * 1388500 = 2500 - 138.85 = 2361.15
```

The gradient is positive because \\(w\\) is slightly too large at some points. The update decreases \\(w\\), moving toward the minimum.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** By hand, compute MSE for 3 data points: \\(x=[1,2,3], y=[2,4,6], w=2, b=0\\). What is \\(\hat{y}\\) for each point? What is the MSE?

{{< pyrepl code="import numpy as np\n\nx = np.array([1.0, 2.0, 3.0])\ny = np.array([2.0, 4.0, 6.0])\nw = 2.0\nb = 0.0\n\n# TODO: compute y_hat and MSE\n# expected: y_hat = [2, 4, 6], MSE = 0.0 (perfect fit)\ny_hat = None\nmse   = None\nprint('y_hat:', y_hat)\nprint('MSE:', mse)" height="180" >}}

2. **Coding:** Use `sklearn.linear_model.LinearRegression` to fit the same sizes/prices data. Compare its `coef_` and `intercept_` to your manually computed \\(w\\) and \\(b\\) after 1000 gradient steps.
3. **Challenge:** Implement full gradient descent (not just one step) on the sizes/prices dataset. Run for 1000 steps with lr=0.0001. Plot MSE over iterations. How many steps until it converges?
4. **Variant:** Try learning rates `lr = 1e-3, 1e-4, 1e-5`. For each, run 100 steps and print the final MSE. What is the best learning rate for this data?
5. **Debug:** The code below has a learning rate that is way too large, causing divergence. Find the line and fix it.

{{< pyrepl code="import numpy as np\n\nx = np.array([1.0, 2.0, 3.0, 4.0, 5.0])\ny = np.array([2.0, 4.0, 6.0, 8.0, 10.0])\n\nw, b = 0.0, 0.0\nlr = 10.0  # BUG: learning rate is far too large\n\nfor step in range(5):\n    y_hat  = w * x + b\n    errors = y - y_hat\n    grad_w = (-2 / len(x)) * np.sum(x * errors)\n    grad_b = (-2 / len(x)) * np.sum(errors)\n    w = w - lr * grad_w\n    b = b - lr * grad_b\n    print(f'Step {step}: w={w:.3f}, b={b:.3f}, MSE={np.mean(errors**2):.3f}')\n\n# TODO: change lr to something sensible so w converges to ~2.0" height="220" >}}

6. **Conceptual:** Why is MSE a better loss function than mean absolute error (MAE) for gradient descent? What property of the squared term makes it easier to optimise?
7. **Recall:** Write the formula for MSE and the gradient \\(\partial MSE / \partial w\\) from memory. Include all terms.
