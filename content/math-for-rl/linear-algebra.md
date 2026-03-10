---
title: "Linear Algebra"
description: "Vectors, dot product, matrices, matrix-vector product, and gradients — with RL motivation and practice."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["linear algebra", "vectors", "matrices", "gradients", "math for RL"]
keywords: ["linear algebra for RL", "vectors", "dot product", "matrix-vector", "gradients"]
---

This page covers the linear algebra you need for RL: vectors, dot products, matrices, matrix-vector multiplication, and the idea of gradients. [Back to Math for RL](../).

---

## Core concepts

### Vectors

A **vector** is an ordered list of numbers, e.g. \\(x = [x_1, x_2, x_3]^T\\) (column vector). We treat it as a column by default. The **dot product** of two vectors \\(x\\) and \\(y\\) of the same length is \\(x^T y = \sum_i x_i y_i\\). Geometrically, it is related to the angle between the vectors and their lengths: \\(x^T y = \|x\| \|y\| \cos\theta\\).

**In reinforcement learning:** A **state** or **observation** is often represented as a vector (e.g. position and velocity in CartPole: 4 numbers). Linear value approximation uses \\(V(s) \approx w^T x(s)\\), where \\(x(s)\\) is a feature vector for state \\(s\\) and \\(w\\) is a weight vector. The dot product \\(w^T x(s)\\) is the predicted value.

### Matrices and matrix-vector product

A **matrix** \\(A\\) has rows and columns. The product \\(A w\\) (matrix times column vector \\(w\\)) is a new vector whose \\(i\\)-th entry is the dot product of the \\(i\\)-th row of \\(A\\) with \\(w\\). So \\(A w\\) has shape (number of rows of \\(A\\), 1).

**In reinforcement learning:** In linear function approximation, we often have a design matrix \\(X\\) (one row per state or transition) and weights \\(w\\); then \\(X w\\) gives a vector of predicted values. Neural networks are stacks of linear maps (matrix-vector products) plus nonlinearities.

### Gradients

The **gradient** of a scalar function \\(f(w)\\) with respect to a vector \\(w\\) is the vector of partial derivatives: \\(\nabla_w f = \bigl[\frac{\partial f}{\partial w_1}, \ldots, \frac{\partial f}{\partial w_n}\bigr]^T\\). It points in the direction of steepest increase. For \\(f(w) = a^T w\\) (linear), \\(\nabla_w f = a\\). For \\(f(w) = w^T A w\\) (quadratic), \\(\nabla_w f = (A + A^T) w\\); if \\(A\\) is symmetric, \\(\nabla_w f = 2 A w\\).

**Useful fact:** If \\(y = A w\\) and \\(A\\) is constant, then \\(\nabla_w (A w) = A^T\\) (in the convention that gradient is a column vector). So for a scalar loss \\(L = g(A w)\\), the chain rule gives \\(\nabla_w L = A^T \nabla_y g\\).

**In reinforcement learning:** We update weight vectors using gradients: \\(w \leftarrow w + \alpha \nabla_w J\\) (for maximization) or \\(w \leftarrow w - \alpha \nabla_w L\\) (for minimization). Policy gradients and value-function fitting all rely on computing gradients with respect to parameters.

---

## Practice questions

1. **Dot product:** Given \\(x = [1, 2, 3]^T\\) and \\(y = [4, 5, 6]^T\\), compute \\(x^T y\\). What is the geometric interpretation?

{{< collapse summary="Answer and explanation" >}}
**Step 1 — Compute:** \\(x^T y = 1\cdot 4 + 2\cdot 5 + 3\cdot 6 = 4 + 10 + 18 = 32\\).

**Geometric interpretation:** The dot product equals \\(\|x\| \|y\| \cos\theta\\), where \\(\theta\\) is the angle between the vectors. So it measures how much the vectors point in the same direction, scaled by their lengths. If perpendicular, dot product is 0; if parallel, it’s the product of their magnitudes.

**Explanation:** In RL, \\(V(s) = w^T x(s)\\) is a dot product between the weight vector and the feature vector—the value is high when \\(w\\) and \\(x(s)\\) align.

**Python:** `np.dot([1,2,3], [4,5,6])` or `(np.array([1,2,3]) * np.array([4,5,6])).sum()` gives 32.
{{< /collapse >}}

The chart below shows the contribution of each term \\(x_i y_i\\) to the dot product; the sum is 32.

{{< chart type="bar" title="Dot product xᵀy: contribution per dimension (sum = 32)" labels="x₁y₁, x₂y₂, x₃y₃" data="4, 10, 18" >}}

---

2. **Matrix-vector:** Let \\(A = \begin{bmatrix} 1 & 0 \\ 2 & 1 \\ 0 & 1 \end{bmatrix}\\) and \\(w = [1, -1]^T\\). Compute \\(A w\\).

{{< collapse summary="Answer and explanation" >}}
\\(A w\\) is a 3×1 vector; each entry is the dot product of a row of \\(A\\) with \\(w\\).

**Step 1 — Row 1:** \\(1\cdot 1 + 0\cdot(-1) = 1\\).  
**Step 2 — Row 2:** \\(2\cdot 1 + 1\cdot(-1) = 2 - 1 = 1\\).  
**Step 3 — Row 3:** \\(0\cdot 1 + 1\cdot(-1) = -1\\).

**Answer:** \\(A w = [1, 1, -1]^T\\).

**Explanation:** Matrix-vector multiplication is “one dot product per row of \\(A\\)”. In linear function approximation, each row might be one state’s feature vector, and \\(Aw\\) is the vector of predicted values.

**Python:** `A = np.array([[1,0],[2,1],[0,1]]); w = np.array([1,-1]); A @ w` gives `array([1, 1, -1])`.
{{< /collapse >}}

The chart below shows the result \\(A w = [1, 1, -1]^T\\) (one value per row of \\(A\\)).

{{< chart type="bar" title="Aw (matrix-vector product)" labels="Row 1, Row 2, Row 3" data="1, 1, -1" >}}

---

3. **Gradient:** If \\(f(w) = a^T w\\) with \\(a = [1, 2, 3]^T\\), what is \\(\nabla_w f\\)? If \\(y = A w\\) and \\(A\\) is constant, what is \\(\nabla_w (A w)\\) (as a column vector)?

{{< collapse summary="Answer and explanation" >}}
**Part 1:** For \\(f(w) = a^T w\\), we have \\(\frac{\partial f}{\partial w_i} = a_i\\), so \\(\nabla_w f = a = [1, 2, 3]^T\\). The gradient of a linear function is the coefficient vector.

**Part 2:** \\(\nabla_w (A w) = A^T\\) (when the gradient is defined as a column vector). So the gradient of \\(Aw\\) with respect to \\(w\\) is the transpose of \\(A\\). For a scalar loss \\(L = g(Aw)\\), the chain rule gives \\(\nabla_w L = A^T \nabla_y g\\) where \\(y = Aw\\).

**Explanation:** In RL, when we backpropagate through a linear layer \\(y = Aw\\), the gradient with respect to \\(w\\) is \\(A^T\\) times the gradient with respect to \\(y\\).

**Python (gradient of \\(a^T w\\)):** `a = np.array([1.,2.,3.]); f = lambda w: np.dot(a,w);` numerical gradient at `w=np.zeros(3)` is `a`. For \\(Aw\\): `A.T` is the gradient of \\(Aw\\) w.r.t. \\(w\\) (as a column).
{{< /collapse >}}

For \\(f(w) = a^T w\\) with \\(a = [1,2,3]^T\\), the gradient is constant \\(a\\). The chart below shows \\(\nabla_w f = a\\) (one component per dimension).

{{< chart type="bar" title="∇f = a (gradient of aᵀw)" labels="w₁, w₂, w₃" data="1, 2, 3" >}}

---

4. **NumPy:** Create vectors \\(x\\) and \\(y\\) as NumPy arrays and compute their dot product with `np.dot(x, y)`. Create a 2×2 matrix \\(A\\) and a 2-vector \\(w\\), then compute \\(A w\\).

{{< collapse summary="Answer and explanation" >}}
```python
import numpy as np
x = np.array([1., 2., 3.])
y = np.array([4., 5., 6.])
dot = np.dot(x, y)   # 32.0

A = np.array([[1., 0.], [2., 1.], [0., 1.]])
w = np.array([1., -1.])
Aw = A @ w          # or np.dot(A, w)  -> array([ 1.,  1., -1.])
```

**Explanation:** `np.dot(x, y)` gives the dot product for 1D arrays. `A @ w` (or `np.dot(A, w)`) does matrix-vector multiplication. In RL, states are often NumPy arrays and linear layers compute such products.
{{< /collapse >}}

Running the code gives dot product 32 and \\(Aw = [1, 1, -1]^T\\). The chart below shows the entries of \\(Aw\\) (same as question 2).

{{< chart type="bar" title="Result of A @ w (NumPy)" labels="Index 0, Index 1, Index 2" data="1, 1, -1" >}}

---

5. **RL:** In linear value approximation \\(V(s) = w^T x(s)\\), if the true return for a state is \\(G\\) and we use squared-error loss \\((G - w^T x(s))^2\\), write the gradient of this loss with respect to \\(w\\) in one line (using \\(x = x(s)\\)).

{{< collapse summary="Answer and explanation" >}}
Let \\(L = (G - w^T x)^2\\). The scalar \\(w^T x\\) has gradient \\(\nabla_w (w^T x) = x\\). By the chain rule, \\(\frac{\partial L}{\partial (w^T x)} = 2(G - w^T x)\cdot (-1) = -2(G - w^T x)\\), so:

\\(\nabla_w L = -2(G - w^T x)\, x\\).

**Explanation:** This is the gradient used in linear TD and Monte Carlo value prediction: we update \\(w\\) in the direction that reduces the squared error between the predicted value \\(w^T x\\) and the target \\(G\\).

**Python (conceptual):** With `x`, `G`, and predicted `v = np.dot(w,x)`, the gradient is `-2*(G - v)*x`. We then do `w -= alpha * grad` (gradient descent).
{{< /collapse >}}

The gradient \\(\nabla_w L = -2(G - w^T x)x\\) points along the feature vector \\(x\\); its magnitude scales with the TD error \\(|G - w^T x|\\). The chart below shows the two components of \\(\nabla_w L\\) for a 2D example (\\(x=[1,0.5]\\), \\(G - w^T x = 0.5\\)).

{{< chart type="bar" title="∇L components (2D example)" labels="w₁, w₂" data="-1, -0.5" >}}

---

6. **By hand:** For \\(f(w) = w_1^2 + w_2^2\\), compute \\(\nabla_w f\\) at \\(w = (1, 2)\\). In which direction does \\(f\\) increase fastest?

{{< collapse summary="Answer and explanation" >}}
**Step 1:** \\(\frac{\partial f}{\partial w_1} = 2w_1\\), \\(\frac{\partial f}{\partial w_2} = 2w_2\\). So \\(\nabla_w f = [2w_1, 2w_2]^T\\).

**Step 2 — At \\(w = (1, 2)\\):** \\(\nabla_w f = [2, 4]^T\\).

**Direction of steepest increase:** The gradient \\(\nabla f\\) points in the direction of steepest *increase*. So \\(f\\) increases fastest in the direction \\([2, 4]^T\\) (or any positive scalar multiple of it).

**Explanation:** Gradient descent *minimizes* by moving in the direction \\(-\nabla f\\). Here we’re just identifying the gradient and its geometric meaning.
**Python:** `w = np.array([1., 2.]); grad = 2*w` gives `[2. 4.]`.
{{< /collapse >}}

At \\(w = (1, 2)\\) the gradient is \\([2, 4]^T\\). The chart below shows these two components.

{{< chart type="bar" title="∇f at w=(1,2): [2, 4]ᵀ" labels="∂f/∂w₁, ∂f/∂w₂" data="2, 4" >}}

---

7. **RL:** In **robot navigation**, the state might be a 4-vector (x, y, vx, vy). What is the dimension of \\(w\\) in \\(V(s) = w^T \\phi(s)\\) if \\(\phi(s)\\) is the identity (so \\(\phi(s) = s\\))?

{{< collapse summary="Answer and explanation" >}}
If \\(\phi(s) = s\\) and \\(s\\) is a 4-vector, then \\(\phi(s)\\) has dimension 4. For the dot product \\(w^T \phi(s)\\) to be defined, \\(w\\) must have the same length as \\(\phi(s)\\). So \\(w\\) has dimension **4**.

**Explanation:** In linear value approximation, the weight vector has the same dimension as the feature vector. With identity features, the state itself is the feature vector, so \\(w\\) is 4-dimensional.

**Python:** `s = np.array([x, y, vx, vy])  # shape (4,)`; `w = np.zeros(4)` or `np.random.randn(4)`; `V = np.dot(w, s)`. So `w.shape == (4,)`.
{{< /collapse >}}

With \\(\phi(s) = s\\) and state dimension 4, \\(w\\) has 4 components. The chart below shows that state dimension and weight dimension match (both 4).

{{< chart type="bar" title="Dimension of s and w (both 4)" labels="State dim, Weight dim" data="4, 4" >}}

---

## Professor's hints

- In RL papers, “state” often means a vector; “state space” can be discrete (set of indices) or continuous (e.g. \\(\mathbb{R}^n\\)). Linear algebra applies when states or features are vectors.
- When you see \\(\nabla_\theta\\) in RL, \\(\theta\\) is usually the parameter vector (or all parameters) of a neural network or linear approximator. PyTorch and TensorFlow compute these gradients automatically via backprop.
- Row vs column convention: many texts write gradients as rows; others as columns. What matters is that the update \\(w \leftarrow w + \alpha \cdot \text{(gradient)}\\) uses the same shape for \\(w\\) and the gradient.

---

## Common pitfalls

- **Wrong shape in matrix-vector product:** \\(A w\\) requires that the number of *columns* of \\(A\\) equals the length of \\(w\\). The result has length = number of *rows* of \\(A\\).
- **Transpose confusion:** \\(\nabla_w (A w) = A^T\\) (not \\(A\\)) when the gradient is a column vector. Check your convention against the framework you use (PyTorch, etc.).
- **Axis in NumPy:** When you sum or average over “all samples” in a batch, that is usually `axis=0` (over rows). When you want one value per sample, you often use `axis=1`. Always check the shape after the operation.
