---
title: "Linear Algebra"
description: "Vectors, dot product, matrix-vector product, and gradients — with RL motivation and explained solutions."
date: 2026-03-10T00:00:00Z
draft: false
---

This page covers the linear algebra you need for the preliminary assessment: dot product, matrix-vector multiplication, and gradients with respect to vectors. [Back to Preliminary](../).

---

## Why this matters for RL

States and observations are often vectors; linear value approximation uses \\(V(s) \approx w^T x(s)\\); neural networks are built from matrix-vector products and gradients. You need to compute dot products and \\(\nabla_w (Aw)\\) by hand and understand their geometric meaning.

### Learning objectives

Compute dot products and matrix-vector products; state \\(\nabla_w (Aw) = A^T\\) (for column gradient); relate these to state vectors and value approximation.

---

## Core concepts

- Dot product: \\(x^T y = \sum_i x_i y_i\\). Geometrically: \\(x^T y = \|x\| \|y\| \cos\theta\\) (related to angle and lengths).
- Matrix-vector product: \\((Aw)_i\\) = dot product of row \\(i\\) of \\(A\\) with \\(w\\).
- Gradient: \\(\nabla_w (Aw) = A^T\\) when \\(A\\) is constant and gradient is taken as column vector. For a scalar loss \\(L = f(Aw)\\), chain rule gives \\(\nabla_w L = A^T \nabla_y f\\) where \\(y = Aw\\).

---

## Worked problems (with explanations)

### 1. Dot product (Q3)

Q: Given \\(x = [1,2,3]^T\\) and \\(y = [4,5,6]^T\\), compute their dot product. What is the geometric interpretation?

{{< collapse summary="Answer and explanation" >}}
Dot product = \\(1\cdot 4 + 2\cdot 5 + 3\cdot 6 = 4 + 10 + 18 = 32\\).

### Geometric interpretation

The dot product equals \\(\|x\| \|y\| \cos\theta\\), where \\(\theta\\) is the angle between the vectors. So it measures “how much” the vectors point in the same direction, scaled by their lengths. If they are perpendicular, the dot product is 0; if parallel, it is the product of their magnitudes. In RL, when we write \\(V(s) = w^T x(s)\\), we are computing a dot product between the weight vector and the feature vector—the value is high when \\(w\\) and \\(x(s)\\) align.
{{< /collapse >}}

---

### 2. Gradient of \\(Aw\\) (Q4)

Q: If \\(A\\) is a matrix and \\(w\\) is a weight vector, what is \\(\nabla_w (A w)\\)? Assume \\(A\\) is constant.

{{< collapse summary="Answer and explanation" >}}
\\(\nabla_w (A w) = A^T\\) (when the gradient is defined as a column vector of partial derivatives).

For a scalar loss \\(L = f(Aw)\\), the chain rule gives \\(\nabla_w L = A^T \nabla_y f\\), where \\(y = Aw\\) and \\(\nabla_y f\\) is the gradient of \\(f\\) with respect to \\(y\\).

### Explanation

Each component of \\(Aw\\) is a linear function of \\(w\\) (the \\(i\\)-th component is the \\(i\\)-th row of \\(A\\) dotted with \\(w\\)). So \\(\frac{\partial (Aw)_i}{\partial w_j} = A_{ij}\\). Stacking these as a column vector (gradient w.r.t. \\(w\\)) gives the \\(j\\)-th column of the Jacobian, which is the \\(j\\)-th row of \\(A\\)—hence \\(A^T\\). In RL, when we backpropagate through a linear layer \\(y = Aw\\), the gradient with respect to \\(w\\) is \\(A^T\\) times the gradient with respect to \\(y\\).
{{< /collapse >}}

---

### 3. Matrix-vector product by hand

Q: Let \\(A = \begin{bmatrix} 1 & 0 \\\\ 2 & 1 \\\\ 0 & 1 \end{bmatrix}\\) and \\(w = [1, -1]^T\\). Compute \\(A w\\).

{{< collapse summary="Answer and explanation" >}}
\\(Aw\\) is a 3×1 vector. Row 1: \\(1\cdot 1 + 0\cdot(-1) = 1\\). Row 2: \\(2\cdot 1 + 1\cdot(-1) = 1\\). Row 3: \\(0\cdot 1 + 1\cdot(-1) = -1\\). So \\(Aw = [1, 1, -1]^T\\).

### Explanation

Matrix-vector multiplication is “one dot product per row of \\(A\\)”. The result has as many entries as \\(A\\) has rows. In linear function approximation, each row of \\(A\\) might correspond to one state’s feature vector, and \\(Aw\\) would be the vector of predicted values for those states.
{{< /collapse >}}

---

## Math example: why \\(\nabla_w (Aw) = A^T\\)

Let \\(A\\) be \\(m \times n\\) and \\(w\\) be \\(n \times 1\\). Then \\(y = Aw\\) is \\(m \times 1\\), with \\(y_i = \sum_j A_{ij} w_j\\). So \\(\frac{\partial y_i}{\partial w_j} = A_{ij}\\). The gradient of \\(y\\) with respect to \\(w\\) (column vector) has \\(j\\)-th entry \\(\frac{\partial (\text{scalar})}{\partial w_j}\\); for the full vector \\(y\\), the Jacobian \\(\frac{\partial y}{\partial w}\\) has \\((i,j)\\) entry \\(A_{ij}\\), so it is the matrix \\(A\\). The *gradient* of a scalar \\(f(y)\\) w.r.t. \\(w\\) is \\(\frac{\partial y}{\partial w}^T \nabla_y f = A^T \nabla_y f\\). So when we “pull back” the gradient from \\(y\\) to \\(w\\), we multiply by \\(A^T\\). That is exactly what backprop does through a linear layer.

---

## Code example (with explanation)

```python
import numpy as np
x = np.array([1., 2., 3.])
y = np.array([4., 5., 6.])
dot = np.dot(x, y)   # 32.0

A = np.array([[1., 0.], [2., 1.], [0., 1.]])
w = np.array([1., -1.])
Aw = A @ w           # or np.dot(A, w)  -> array([ 1.,  1., -1.])
```

### Explanation

`np.dot(x, y)` computes the dot product for 1D arrays. `A @ w` (or `np.dot(A, w)`) performs matrix-vector multiplication: each row of \\(A\\) is dotted with \\(w\\). In RL, states are often NumPy arrays and linear layers compute exactly such products; PyTorch tensors use the same idea with `torch.mv(A, w)` or `A @ w`.
---

## Professor's hints

- In RL, “state” is often a vector; \\(V(s) = w^T x(s)\\) is a dot product. The gradient of a squared error \\((G - w^T x)^2\\) w.r.t. \\(w\\) is \\(-2(G - w^T x)\, x\\).
- \\(\nabla_w (Aw) = A^T\\) (not \\(A\\)) when the gradient is a column. Check your framework’s convention.
- Matrix-vector shape: \\(A\\) is (rows × cols); \\(w\\) length = cols; \\(Aw\\) length = rows.

---

## Common pitfalls

- Wrong shape: \\(Aw\\) requires columns of \\(A\\) = length of \\(w\\). Result length = rows of \\(A\\).
- Transpose confusion: Gradient w.r.t. \\(w\\) for \\(y = Aw\\) is \\(A^T\\) times gradient w.r.t. \\(y\\), not \\(A\\).
- Axis in NumPy: Summing over samples in a batch is usually `axis=0`; per-sample quantities often use `axis=1`. Check shapes after operations.
