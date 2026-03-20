---
title: "NumPy"
description: "Arrays, indexing, slicing, and element-wise vs matrix operations — with RL-relevant examples and explanations."
date: 2026-03-10T00:00:00Z
draft: false
difficulty: 3
tags: ["NumPy", "arrays", "indexing", "slicing", "preliminary"]
keywords: ["NumPy for RL", "arrays", "indexing slicing", "element-wise matrix operations"]
weight: 2
roadmap_icon: "layers"
roadmap_color: "teal"
roadmap_phase_label: "Topic 2 · NumPy"
---

This page covers the NumPy you need for the preliminary assessment: creating arrays, indexing, slicing, and element-wise operations. [Back to Preliminary](../).

---

## Why this matters for RL

Environments return observations as arrays; neural networks consume batches of arrays. NumPy is the standard way to represent states, reward vectors, and batches of transitions. You need to create and reshape arrays, slice them, and know the difference between element-wise and matrix multiplication.

### Learning objectives

Create and index NumPy arrays; set rows/columns; compute element-wise products and matrix-vector products; use `np.dot` or `@` correctly.

---

## Core concepts

- Creation: `np.zeros(shape)`, `np.array(list)`. Shape is a tuple, e.g. `(3, 3)`.
- Indexing: `arr[0]` is the first row; `arr[i, j]` is element at row `i`, column `j`. Slicing: `arr[i:j]`, `arr[:, j]`.
- Element-wise: `arr * arr`, `arr + 1`, `np.square(arr)`. Same shape in and out.
- Matrix product: `A @ b` or `np.dot(A, b)` for matrix-vector; shapes must match (cols of A = length of b).

---

## Worked problems (with explanations)

### 1. 3×3 array and element-wise product (Q8)

Q: Create a 3×3 NumPy array of zeros, then set the first row to [1,2,3]. How do you compute the element-wise product of this array with itself?

{{< collapse summary="Answer and explanation" >}}
```python
import numpy as np
arr = np.zeros((3, 3))
arr[0] = [1, 2, 3]
prod = arr * arr   # or np.square(arr)
```

After setting the first row, `arr` is  
[[1, 2, 3],  
 [0, 0, 0],  
 [0, 0, 0]].  
Then `arr * arr` squares each entry element-wise: [[1, 4, 9], [0, 0, 0], [0, 0, 0]].

### Explanation

`np.zeros((3,3))` allocates a 3×3 array of zeros. Assigning `arr[0] = [1,2,3]` fills the first row. The `*` operator is element-wise; for matrix multiplication we would use `arr @ arr` (or `np.dot(arr, arr)`), which would be a 3×3 matrix product. In RL we often do element-wise ops on reward masks or squared errors, and matrix products for linear layers.
{{< /collapse >}}

---

### 2. Dot product and matrix-vector product

Q: Create vectors `x` and `y` as NumPy arrays of length 3 and compute their dot product. Create a 2×3 matrix `A` and a 3-vector `w`; compute `A @ w`.

{{< collapse summary="Answer and explanation" >}}
```python
import numpy as np
x = np.array([1., 2., 3.])
y = np.array([4., 5., 6.])
dot = np.dot(x, y)   # or x @ y  -> 32.0

A = np.array([[1., 0., 1.], [0., 1., 1.]])  # 2×3
w = np.array([1., -1., 2.])
Aw = A @ w   # or np.dot(A, w)  -> array([3., 1.])
```

### Explanation

For 1D arrays, `np.dot(x, y)` is the dot product (a scalar). For matrix-vector, `A @ w` requires the number of columns of `A` (3) to equal the length of `w` (3); the result has length equal to the number of rows of `A` (2). Each entry of `Aw` is the dot product of one row of `A` with `w`. In linear value approximation, rows of `A` could be feature vectors and `Aw` the predicted values.
{{< /collapse >}}

---

### 3. Slicing and shape

Q: Create a 4×5 array of consecutive integers (0 to 19). Extract the 2×2 block in the top-left corner. Extract the last column.

{{< collapse summary="Answer and explanation" >}}
```python
arr = np.arange(20).reshape(4, 5)
top_left = arr[:2, :2]   # rows 0–1, cols 0–1
last_col = arr[:, -1]    # all rows, last column
```

### Explanation

`reshape(4, 5)` gives a 4×5 array. Slicing `arr[:2, :2]` takes the first two rows and first two columns. `arr[:, -1]` takes all rows and the last column (index -1). In RL, we often slice batches: e.g. `states[batch_inds]` or `rewards[:, -1]` for the last step rewards.
{{< /collapse >}}

---

## Code examples (with explanations)

### Batch of states and mean

```python
import numpy as np
# 10 “states,” each of dimension 4 (e.g. CartPole)
states = np.random.randn(10, 4)
mean_state = states.mean(axis=0)   # shape (4,)
std_state = states.std(axis=0)     # shape (4,)
```

### Explanation

`axis=0` means “along the first dimension,” so we average over the 10 samples and get one vector of length 4. In RL we often normalize state batches this way; `mean(axis=0)` and `std(axis=0)` are used in running mean/std normalization for observations.
---

## Professor's hints

- Always check shapes: `arr.shape` after creation and after each operation. Matrix product: (m,n) @ (n,) → (m,).
- Element-wise `*` requires same shape (or broadcastable). For dot product of two 1D arrays use `np.dot(a, b)` or `a @ b`.
- Use `arr.reshape(-1, n)` to flatten leading dimensions and keep `n` columns; NumPy infers the first dimension.

---

## Common pitfalls

- Confusing `*` and `@`: `A * B` is element-wise; `A @ B` is matrix product. Wrong choice gives wrong shape or wrong numbers.
- Axis: `axis=0` is “over rows” (first index); `axis=1` is “over columns.” Sum/mean over samples in a batch is usually `axis=0`.
- Integer array indexing: `arr[0]` gives a view when possible; assigning to a slice (e.g. `arr[0] = x`) modifies the original array.
