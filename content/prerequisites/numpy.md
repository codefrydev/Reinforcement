---
title: "NumPy"
description: "NumPy for RL: arrays, indexing, broadcasting, random, and batch operations."
date: 2026-03-10T00:00:00Z
weight: 20
draft: false
---

Used in [Preliminary: NumPy](../preliminary/numpy/) and throughout the curriculum for state/observation arrays, reward vectors, and batch operations. RL environments return observations as arrays; neural networks consume batches of arrays—NumPy is the standard bridge.

---

## Why NumPy matters for RL

- **Arrays** — States and observations are vectors or matrices; rewards over time are 1D arrays. `np.zeros()`, `np.array()`, `np.arange()` are used constantly.
- **Indexing and slicing** — Extract rows/columns, mask by condition, gather batches. Fancy indexing appears in replay buffers and minibatches.
- **Broadcasting** — Apply operations across shapes without writing loops (e.g. subtract mean from a batch).
- **Random** — `np.random` for \\(\epsilon\\)-greedy, environment stochasticity, and reproducible seeds.
- **Math** — `np.sum`, `np.mean`, dot products, element-wise ops. No need for Python loops over elements.

---

## Core concepts with examples

### Creating arrays

```python
import numpy as np

# Preallocate for states (e.g. 4D state for CartPole)
state = np.zeros(4)
state = np.array([0.1, -0.2, 0.05, 0.0])

# Grid of values (e.g. for value function over 2D grid)
grid = np.zeros((3, 3))
grid[0] = [1, 2, 3]

# Ranges and linspace
steps = np.arange(0, 1000, 1)       # 0, 1, ..., 999
x = np.linspace(0, 1, 11)           # 11 points from 0 to 1
```

### Shape, reshape, and batch dimension

```python
arr = np.array([[1, 2], [3, 4], [5, 6]])  # shape (3, 2)
batch = arr.reshape(1, 3, 2)               # (1, 3, 2) for "1 sample"
flat = arr.flatten()                       # (6,)
```

### Indexing and slicing

```python
# Slicing: first two rows, all columns
arr[:2, :]

# Last row
arr[-1, :]

# Boolean mask: rows where first column > 2
mask = arr[:, 0] > 2
arr[mask]

# Integer indexing: rows 0 and 2
arr[[0, 2], :]
```

### Broadcasting and element-wise ops

```python
# Subtract mean from each column
X = np.random.randn(32, 4)  # 32 samples, 4 features
X_centered = X - X.mean(axis=0)

# Element-wise product (e.g. importance weights)
a = np.array([1.0, 2.0, 0.5])
b = np.array([1.0, 1.0, 2.0])
a * b   # array([1., 2., 1.])
```

### Random and seeding

```python
np.random.seed(42)
# Unit Gaussian (for bandit rewards, noise)
samples = np.random.randn(10)
# Uniform [0, 1)
u = np.random.rand(5)
# Random integers in [low, high)
action = np.random.randint(0, 4)  # one of 0,1,2,3
```

### Useful reductions

```python
arr = np.array([[1, 2], [3, 4], [5, 6]])
arr.sum()           # 21
arr.sum(axis=0)     # [9, 12]
arr.mean(axis=1)    # [1.5, 3.5, 5.5]
np.max(arr, axis=0) # [5, 6]
```

---

## Exercises

**Exercise 1.** Create a 3×3 NumPy array of zeros, set the first row to `[1, 2, 3]`, and compute the element-wise product of the array with itself. Then compute the sum of all elements (should be 1²+2²+3² = 14).

**Exercise 2.** Given a 1D array `rewards` of length 100 (e.g. `rewards = np.random.rand(100)`), compute the **discounted cumulative return** from each starting index: for index \\(t\\), compute \\(G_t = \\sum_{k=0}^{T-t} \\gamma^k r_{t+k}\\) with \\(\gamma=0.99\\). Use a loop over \\(t\\) and inner sum over \\(k\\); return a 1D array of length 100.

**Exercise 3.** Create a (32, 4) array of standard normal samples. Compute the mean and standard deviation along axis 0 (so you get 4 means and 4 stds). Then normalize the array to zero mean and unit variance per column.

**Exercise 4.** Use `np.random.randint(0, 10, size=(5, 5))` to get a 5×5 matrix of random integers in \\([0, 10)\\). Find the indices (row, col) of the maximum value using `np.unravel_index` and `np.argmax`.

**Exercise 5.** Implement a simple replay buffer as a circular buffer: preallocate `states = np.zeros((capacity, state_dim))` and `rewards = np.zeros(capacity)`. Write functions `push(s, r)` and `sample(batch_size)` that store the latest transition and return a random batch of indices, then return `states[indices]` and `rewards[indices]`. Assume `state_dim=4` and `capacity=100` for testing.

**Exercise 6.** Create a 1D array of 100 zeros. Set every 10th element (indices 0, 10, 20, …) to 1. Compute the mean of the array. **In RL:** This mimics sparse reward signals; many steps give 0, a few give 1.

**Exercise 7.** Given `rewards = np.array([0.0, 0.0, 1.0])` and `gamma = 0.9`, compute the discounted return \\(G_0 = r_0 + \\gamma r_1 + \\gamma^2 r_2\\) using NumPy (e.g. `np.sum(gamma ** np.arange(3) * rewards)`). Verify you get 0.81.

**Exercise 8.** (Challenge) Build a (1000, 4) array of random states (e.g. `np.random.randn(1000, 4)`). Normalize each row to unit length (divide each row by its L2 norm). Check that `np.linalg.norm(arr, axis=1)` is all ones. **In RL:** State normalization can stabilize learning.

---

## Professor's hints

- **Set `np.random.seed(42)`** (or any fixed number) at the start of your script when debugging; RL results are reproducible then, and you can compare runs.
- Use **axis=0** for "over rows" (e.g. mean over samples in a batch); **axis=1** for "over columns" (e.g. mean over features per sample). When in doubt, check the shape before and after: `arr.shape` → `arr.mean(axis=0).shape`.
- Preallocate arrays with `np.zeros` or `np.empty` when you know the size (e.g. buffers); it is faster than appending in a loop.
- **In RL:** States from Gym are often NumPy arrays; batching for neural networks means stacking states into a 2D array (batch_size, state_dim). Always verify shapes when connecting env and network.

---

## Common pitfalls

- **In-place vs copy:** Operations like `arr += 1` or `arr[0] = 5` modify the original array. Slicing returns a *view* in many cases; modifying the slice modifies the original. Use `arr.copy()` when you need an independent copy.
- **Mixing up axis:** `arr.mean(axis=0)` reduces dimension 0 (rows), so you get one value per column. Wrong axis gives wrong shapes and wrong semantics (e.g. normalizing per sample vs per feature).
- **Integer indexing with lists/arrays:** `arr[[0, 2], :]` selects rows 0 and 2; `arr[0, 2]` is a single element. Using a list/array of indices returns an array; using a single int drops that dimension.
- **Seeding only once:** If you call `np.random.randn()` in a loop and expect reproducibility, seed once at the top. Reseeding every iteration makes the sequence deterministic but identical every time.

---

**Docs:** [numpy.org/doc](https://numpy.org/doc/stable/).
