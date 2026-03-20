---
title: "NumPy"
description: "NumPy for RL: arrays, indexing, broadcasting, random, and batch operations."
date: 2026-03-10T00:00:00Z
weight: 20
draft: false
difficulty: 2
tags: ["NumPy", "arrays", "RL", "prerequisites"]
keywords: ["NumPy for RL", "arrays", "indexing", "matrix operations", "RL examples"]
roadmap_icon: "layers"
roadmap_color: "teal"
roadmap_phase_label: "Phase 2 · NumPy"
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

## Worked examples

**Example 1 — Discounted return (Exercise 7).** Given `rewards = np.array([0.0, 0.0, 1.0])` and `gamma = 0.9`, compute \\(G_0 = r_0 + \\gamma r_1 + \\gamma^2 r_2\\) using NumPy.

{{< collapse summary="Solution" >}}
**Step 1:** Discount factors: `gamma ** np.arange(3)` gives `[1., 0.9, 0.81]`. **Step 2:** Element-wise product with rewards: `(gamma ** np.arange(3)) * rewards` → `[0., 0., 0.81]`. **Step 3:** Sum: `np.sum(...)` → **0.81**. One-liner: `G = np.sum((gamma ** np.arange(len(rewards))) * rewards)`. This is the same formula used for returns in every RL algorithm.
{{< /collapse >}}

**Example 2 — 3×3 array and element-wise product (Exercise 1).** Create a 3×3 array of zeros, set the first row to `[1, 2, 3]`, compute the element-wise product with itself, then the sum of all elements.

{{< collapse summary="Solution" >}}
**Step 1:** `arr = np.zeros((3, 3))` then `arr[0] = [1, 2, 3]` (first row is 1,2,3; rest zeros). **Step 2:** Element-wise product: `prod = arr * arr` → first row becomes [1, 4, 9]; others stay 0. **Step 3:** Sum: `prod.sum()` = 1+4+9 = **14**. In RL we use `arr * arr` for squared errors or masks; `arr @ arr` would be matrix multiplication.
{{< /collapse >}}

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

---

## Additional exercises

### Micro-exercises (do these now)

1. `import numpy as np; a = np.array([1,2,3]); print(a * 2)` — predict the output before running.
2. Create `np.zeros((3,3))` and set the diagonal to 1 using `np.eye(3)`. Print it.
3. `a = np.array([3,1,4,1,5]); print(np.argmax(a))` — what does argmax return?
4. `a = np.array([0.5, 0.3, 0.2]); print(a.sum(), a.mean())` — predict both outputs.
5. `Q = np.zeros((5,4)); Q[2,3] = 1.5; print(Q[2])` — what is printed?

{{< pyrepl code="import numpy as np\n# Try the micro-exercises above\na = np.array([1, 2, 3])\nprint(a * 2)" height="200" >}}

### Build something small

1. Create a 5×5 value grid `V = np.zeros((5,5))`. Set `V[4,4] = 10` (goal). Print the grid.
2. Given Q-values `Q = np.random.seed(42); Q = np.random.randn(4)`, find the best action with `np.argmax(Q)`. Print Q and the best action.
3. Simulate 100 bandit pulls: `rewards = np.random.normal(0.5, 1, 100)`. Compute mean, std, and max.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n# TODO: create 5x5 value grid, set goal, find best action\nV = np.zeros((5, 5))\nV[4, 4] = 10\nprint(V)" height="200" >}}

### Mini-project: Bandit Q-estimates

Simulate a 3-armed bandit for 500 steps using NumPy arrays (no Python lists). True means: `[0.2, 0.7, 0.1]`. Use `np.random.seed(42)`. Track `Q = np.zeros(3)` and `N = np.zeros(3, dtype=int)`. After 500 steps, plot Q estimates vs true means with matplotlib.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\ntrue_means = np.array([0.2, 0.7, 0.1])\nQ = np.zeros(3)\nN = np.zeros(3, dtype=int)\n# TODO: 500 steps of epsilon-greedy (epsilon=0.1)\n# Update Q[a] += (reward - Q[a]) / N[a]\n" height="240" >}}
