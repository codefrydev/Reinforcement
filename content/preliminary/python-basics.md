---
title: "Python basics for RL and the preliminary assessment"
description: "Functions, lists, loops, and list comprehensions — with RL-relevant examples and explained solutions."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["Python", "functions", "loops", "list comprehensions", "preliminary"]
keywords: ["Python basics", "functions lists loops", "list comprehensions", "RL-relevant examples"]
---

This page covers the Python you need for the preliminary assessment: writing functions, working with lists, and using list comprehensions. [Back to Preliminary](../).

---

## Why this matters for RL

RL code is full of trajectories (lists of states, actions, rewards), configs (dicts), and custom types (agents, buffers). You need to write clear functions, slice sequences, and aggregate data. Moving averages and rolling computations appear when processing reward sequences or returns.

### Learning objectives

Write a function that returns the moving average of a list; use list comprehensions and loops; structure code for clarity and reuse.

---

## Core concepts

- Functions: `def name(args):` body with optional `return`. Use default arguments (e.g. `window=3`) for flexibility.
- Lists and slicing: `arr[i:j]` is elements from index `i` up to (not including) `j`. `len(arr)`, `sum(arr)`.
- List comprehensions: `[expr for i in range(n)]` or `[f(x) for x in items]` build lists in one line. Useful for moving windows and transforms.

---

## Worked problems (with explanations)

### 1. Moving average (Q7)

Q: Write a Python function that takes a list of numbers and returns the moving average with window size 3. For example, input [1,2,3,4,5] returns [2.0, 3.0, 4.0].

{{< collapse summary="Answer and explanation" >}}
```python
def moving_average(arr, window=3):
    return [sum(arr[i:i+window]) / window for i in range(len(arr) - window + 1)]
```

For `[1, 2, 3, 4, 5]`: windows are [1,2,3], [2,3,4], [3,4,5]; averages are 2.0, 3.0, 4.0.

### Explanation

We slide a window of length `window` along the list. For each starting index `i`, `arr[i:i+window]` is that slice; we take its mean. The number of such windows is `len(arr) - window + 1`. In RL, you might compute a moving average of episode returns to smooth learning curves or average rewards over a short horizon.
{{< /collapse >}}

The graph below shows the moving average output (2, 3, 4) for the three windows over [1, 2, 3, 4, 5].

{{< chart type="line" palette="return" title="Moving average (window 3) of [1,2,3,4,5]" labels="Avg(1-3), Avg(2-4), Avg(3-5)" data="2, 3, 4" xLabel="Window" yLabel="Average" >}}

---

### 2. List of returns from rewards

Q: Write a function that takes a list of rewards and a discount factor `gamma` and returns the list of returns (each return = sum of discounted future rewards from that time step). Assume a finite list (e.g. one episode).

{{< collapse summary="Answer and explanation" >}}
```python
def returns_from_rewards(rewards, gamma=0.99):
    n = len(rewards)
    out = []
    for t in range(n):
        g = sum(gamma**(k - t) * rewards[k] for k in range(t, n))
        out.append(g)
    return out
```

### Explanation

From time \\(t\\), the return is \\(G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots\\). For each \\(t\\), we sum \\(\gamma^{k-t} r_k\\) for \\(k = t, t+1, \ldots, n-1\\). This is the same idea as “sum of discounted future rewards” in the definition of value functions. In practice we often compute returns backward for efficiency, but this forward version is clear and correct for short episodes.
{{< /collapse >}}

---

### 3. Dict of per-arm sample means

Q: Given a list of (arm_index, reward) pairs, build a dict mapping each arm to the list of rewards observed for that arm. Then write a one-liner that turns that into a dict of arm → sample mean.

{{< collapse summary="Answer and explanation" >}}
```python
data = [(0, 1.2), (1, 0.8), (0, 1.5), (1, 0.3), (0, 2.1)]  # example
arm_rewards = {}
for arm, r in data:
    arm_rewards.setdefault(arm, []).append(r)
# arm_rewards: {0: [1.2, 1.5, 2.1], 1: [0.8, 0.3]}
means = {arm: sum(rewards)/len(rewards) for arm, rewards in arm_rewards.items()}
# means: {0: 1.6, 1: 0.55}
```

### Explanation

We group rewards by arm using a dict of lists. `setdefault(arm, []).append(r)` creates an empty list for a new arm and appends the reward. The dict comprehension then computes the sample mean for each arm. In a bandit algorithm we maintain such estimates and update them as we pull arms.
{{< /collapse >}}

---

## Code examples (with explanations)

### Moving average — loop version

```python
def moving_average_loop(arr, window=3):
    result = []
    for i in range(len(arr) - window + 1):
        window_sum = sum(arr[i:i+window])
        result.append(window_sum / window)
    return result
```

### Explanation

Same logic as the list-comprehension version: for each valid start index, sum the slice and divide by `window`. The loop form is easier to extend (e.g. if you later want to skip invalid windows or add logging). Both are correct; use whichever is clearer for the task.

---

## Professor's hints

- Prefer a small number of well-named functions over one long script. RL codebases have functions for “compute return,” “update Q,” “select action,” etc.
- Use `arr[i:i+window]` for sliding windows; remember that the last valid index for a window of length `w` is `len(arr) - w`.
- List comprehensions are ideal for one-to-one transforms; use a loop when you need to accumulate state (e.g. building a dict of lists).

---

## Common pitfalls

- Off-by-one in range: For moving average with window 3, the last window starts at index `len(arr)-3`, so `range(len(arr) - window + 1)` is correct. Using `range(len(arr))` would go out of bounds on the slice.
- Integer division: In Python 3, `sum(arr)/len(arr)` is float division when `sum` is float or when you use a float in the expression. If both are integers, use `sum(arr)/len(arr)` (still float in 3) or explicitly `float(len(arr))` to avoid truncation.
- Mutating lists while iterating: Don’t append to a list you’re iterating over; build a new list or iterate over a copy.
