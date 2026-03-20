---
title: "Multiple Regression"
description: "Extend linear regression to multiple features using matrix form ŷ = Xw + b and vectorized NumPy operations."
date: 2026-03-20T00:00:00Z
weight: 5
draft: false
difficulty: 4
tags: ["multiple regression", "matrix form", "NumPy", "linear algebra", "ml-foundations"]
keywords: ["multiple linear regression", "matrix multiplication", "NumPy vectorization", "multi-feature regression", "linear value function", "RL state features"]
roadmap_icon: "layers"
roadmap_color: "purple"
roadmap_phase_label: "Chapter 5"
---

**Learning objectives**

- Write the matrix form of linear regression \\(\hat{y} = Xw + b\\) and identify the shape of each term.
- Compute predictions, MSE, and the gradient \\(\nabla_w MSE\\) using NumPy matrix operations.
- Perform one vectorized gradient descent step on a multi-feature dataset.

**Concept and real-world motivation**

Real datasets have many input features, not just one. A house's price depends on size, number of rooms, age, location score, and more. When we have \\(d\\) features, the linear model becomes:

\\[\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b = w^T x + b\\]

For \\(n\\) samples at once, we pack all inputs into a matrix \\(X \in \mathbb{R}^{n \times d}\\) and all weights into a vector \\(w \in \mathbb{R}^d\\):

\\[\hat{y} = Xw + b\\]

The gradient of MSE with respect to \\(w\\) in matrix form is:

\\[\nabla_w MSE = \frac{-2}{n} X^T (y - \hat{y})\\]

This single formula replaces the per-feature loop — one matrix multiply does the work for all \\(d\\) features simultaneously. This is **vectorization**, and it is how all modern ML code achieves speed.

**RL connection:** RL agents often operate in continuous state spaces where the state \\(s\\) is a vector of measurements: position \\((x, y)\\), velocity \\((\dot{x}, \dot{y})\\), angle \\(\theta\\), angular velocity \\(\dot\theta\\), and more. Linear value function approximation computes \\(V(s) = w^T s\\) — which is exactly \\(\hat{y} = Xw\\) (one sample at a time). Even when we move to neural networks, the first and last layers are matrix multiplications of this exact form.

**Illustration:** Multiple features all contribute to a single prediction through a weighted sum.

{{< mermaid >}}
flowchart LR
    x1["Feature 1\n(size)"] --> dot["Weighted sum\nŷ = w₁x₁ + w₂x₂ + w₃x₃ + b"]
    x2["Feature 2\n(rooms)"] --> dot
    x3["Feature 3\n(age)"] --> dot
    dot --> yhat["Prediction ŷ\n(price)"]
{{< /mermaid >}}

Let us first verify matrix multiply by hand for a tiny 2×2 case:

{{< pyrepl code="import numpy as np\n\n# 2 samples, 2 features\nX = np.array([[1.0, 2.0],\n              [3.0, 4.0]])\nw = np.array([0.5, 1.0])  # shape (2,)\n\n# Matrix multiply: X @ w\ny_hat = X @ w   # same as [1*0.5+2*1.0, 3*0.5+4*1.0] = [2.5, 5.5]\nprint('X @ w =', y_hat)         # expected: [2.5 5.5]\nprint('shapes: X=%s, w=%s, y_hat=%s' % (X.shape, w.shape, y_hat.shape))" height="200" >}}

**Exercise:** Given a 5-sample, 3-feature dataset, compute predictions, MSE, the gradient \\(\nabla_w MSE\\), and perform one gradient step — all using NumPy matrix operations.

{{< pyrepl code="import numpy as np\n\n# 5 samples, 3 features: [size, rooms, age]\nX = np.array([\n    [80,  3, 10],\n    [120, 4, 5 ],\n    [60,  2, 20],\n    [150, 5, 2 ],\n    [100, 3, 8 ],\n], dtype=float)\n\ny = np.array([250000, 380000, 180000, 480000, 310000], dtype=float)\n\nw = np.array([2000.0, 15000.0, -500.0])  # initial weights\nb = 50000.0\nlr = 1e-7\n\n# TODO: compute predictions y_hat = X @ w + b\ny_hat = None\n\n# TODO: compute MSE\nmse = None\n\n# TODO: compute gradient of MSE w.r.t. w\n# grad_w = (-2/n) * X.T @ (y - y_hat)\ngrad_w = None\n\n# TODO: update w with one gradient step\nw_new = None\n\nprint('y_hat:  ', y_hat)\nprint('MSE:    ', mse)\nprint('grad_w: ', grad_w)\nprint('w_new:  ', w_new)\n# expected: y_hat should be close to y; MSE near 0 means good initial guess" height="300" >}}

**Professor's hints**

- `X @ w` performs the matrix-vector multiply — NumPy handles all \\(n\\) samples at once.
- `X.T` transposes X — shape goes from `(n, d)` to `(d, n)`.
- `X.T @ (y - y_hat)` is a dot product of each feature column with the residuals — shape `(d,)`, one gradient per weight.
- Check shapes at every step: `y_hat.shape` should be `(5,)`, `grad_w.shape` should be `(3,)`.
- The learning rate `1e-7` is small because the features (sizes in sq metres, prices in dollars) have very different scales.

**Common pitfalls**

- **Shape mismatch:** If `w` has shape `(3, 1)` instead of `(3,)`, `X @ w` returns shape `(5, 1)` instead of `(5,)`, and subsequent operations may behave unexpectedly. Always use 1-D weight vectors: `w = np.array([...])` not `w = np.array([[...]])`.
- **Forgetting to transpose:** `X.T @ residuals` (shape `(d,)`) is correct. `X @ residuals` (shape `(n,)`) is wrong — it computes a residual-weighted combination of feature vectors, not the gradient.
- **Using a loop instead of matrix ops:** Writing `for j in range(d): grad_w[j] = ...` works but is 10–100× slower than `X.T @ residuals`. Always prefer matrix operations.

{{< collapse summary="Worked solution" >}}
Fully vectorized multiple regression step:

```python
import numpy as np

X = np.array([[80,3,10],[120,4,5],[60,2,20],[150,5,2],[100,3,8]], dtype=float)
y = np.array([250000,380000,180000,480000,310000], dtype=float)
w = np.array([2000.0, 15000.0, -500.0])
b = 50000.0
lr = 1e-7
n = len(y)

# Step 1: predictions
y_hat = X @ w + b              # shape (5,)

# Step 2: MSE
mse = np.mean((y - y_hat)**2)

# Step 3: gradient
residuals = y - y_hat          # shape (5,)
grad_w = (-2/n) * (X.T @ residuals)   # shape (3,)
grad_b = (-2/n) * np.sum(residuals)

# Step 4: update
w_new = w - lr * grad_w
b_new = b - lr * grad_b
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Manually compute \\(\hat{y}\\) for sample 1 (size=80, rooms=3, age=10) using weights \\(w=[2000, 15000, -500], b=50000\\). Show each multiplication. Verify with NumPy.
2. **Coding:** Use `sklearn.linear_model.LinearRegression` to fit the same X, y data. Compare `model.coef_` to the weights your gradient descent converges to after 10,000 steps.
3. **Challenge:** Add a fourth feature to X: distance from city centre (in km) = [5, 2, 8, 1, 4]. Re-run gradient descent. Does adding this feature lower the final MSE?
4. **Variant:** Normalise each feature column in X to have mean 0 and std 1 before training. How does this affect the learning rate you need and the convergence speed?
5. **Debug:** The code below has a shape mismatch bug — `w` is defined as a column vector, causing `X @ w` to have shape `(5, 1)` instead of `(5,)`. Fix it.

{{< pyrepl code="import numpy as np\n\nX = np.array([[1.,2.],[3.,4.],[5.,6.]])\ny = np.array([1., 2., 3.])\n\nw = np.array([[0.5], [0.5]])  # BUG: w has shape (2,1) not (2,)\n\ny_hat = X @ w\nprint('y_hat shape:', y_hat.shape)  # prints (3,1) — should be (3,)\n\nmse = np.mean((y - y_hat)**2)       # BUG: broadcasting issue here\nprint('MSE:', mse)\n\n# TODO: fix w so it has shape (2,) and mse computes correctly" height="200" >}}

6. **Conceptual:** Why is the matrix form \\(\hat{y} = Xw + b\\) more useful than writing a separate equation for each sample? What would happen if \\(n = 1{,}000{,}000\\)?
7. **Recall:** Write the matrix gradient formula \\(\nabla_w MSE\\) from memory. State the shape of \\(X\\), \\(w\\), and the gradient.
