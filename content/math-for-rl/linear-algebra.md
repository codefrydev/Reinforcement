---
title: "Linear Algebra"
description: "Vectors, dot product, matrices, matrix-vector product, and gradients — with RL motivation and practice."
date: 2026-03-10T00:00:00Z
draft: false
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
2. **Matrix-vector:** Let \\(A = \begin{bmatrix} 1 & 0 \\ 2 & 1 \\ 0 & 1 \end{bmatrix}\\) and \\(w = [1, -1]^T\\). Compute \\(A w\\).
3. **Gradient:** If \\(f(w) = a^T w\\) with \\(a = [1, 2, 3]^T\\), what is \\(\nabla_w f\\)? If \\(y = A w\\) and \\(A\\) is constant, what is \\(\nabla_w (A w)\\) (as a column vector)?
4. **NumPy:** Create vectors \\(x\\) and \\(y\\) as NumPy arrays and compute their dot product with `np.dot(x, y)`. Create a 2×2 matrix \\(A\\) and a 2-vector \\(w\\), then compute \\(A w\\).
5. **RL:** In linear value approximation \\(V(s) = w^T x(s)\\), if the true return for a state is \\(G\\) and we use squared-error loss \\((G - w^T x(s))^2\\), write the gradient of this loss with respect to \\(w\\) in one line (using \\(x = x(s)\\)).
6. **By hand:** For \\(f(w) = w_1^2 + w_2^2\\), compute \\(\nabla_w f\\) at \\(w = (1, 2)\\). In which direction does \\(f\\) increase fastest?
7. **RL:** In **robot navigation**, the state might be a 4-vector (x, y, vx, vy). What is the dimension of \\(w\\) in \\(V(s) = w^T \\phi(s)\\) if \\(\phi(s)\\) is the identity (so \\(\phi(s) = s\\))?

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
