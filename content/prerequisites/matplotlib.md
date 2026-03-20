---
title: "Matplotlib"
description: "Matplotlib for RL: learning curves, subplots, heatmaps, and saving figures."
date: 2026-03-10T00:00:00Z
weight: 40
draft: false
difficulty: 2
tags: ["Matplotlib", "plotting", "learning curves", "prerequisites"]
keywords: ["Matplotlib for RL", "learning curves", "subplots", "heatmaps", "saving figures"]
roadmap_icon: "chart"
roadmap_color: "purple"
roadmap_phase_label: "Phase 2 · Matplotlib"
---

Used in many chapter exercises to plot average reward over time, value functions, policy comparisons, and hyperparameter heatmaps. A clear plot often reveals convergence or instability at a glance.

---

## Why Matplotlib matters for RL

- **Line plots** — Reward vs episode, loss vs step, value vs state. The default `plt.plot(x, y)`.
- **Multiple curves** — Overlay several runs or algorithms; use `label` and `legend()`.
- **Subplots** — Several panels in one figure (e.g. reward, length, loss).
- **Heatmaps** — Value function over 2D state space; grid search over \\(\alpha\\) and \\(\epsilon\\).
- **Saving** — `plt.savefig("curve.png", dpi=150)` for reports and slides.

---

## Core concepts with examples

### Single line plot

```python
import matplotlib.pyplot as plt
import numpy as np

episodes = np.arange(100)
rewards = 0.1 * episodes + 0.5 + np.random.randn(100) * 0.5

plt.figure(figsize=(8, 4))
plt.plot(episodes, rewards, alpha=0.7, label="raw")
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title("Learning curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Smoothed curve (moving average)

```python
window = 10
smooth = np.convolve(rewards, np.ones(window)/window, mode="valid")
x_smooth = np.arange(len(smooth))
plt.plot(episodes, rewards, alpha=0.3, label="raw")
plt.plot(x_smooth, smooth, label=f"MA-{window}")
plt.legend()
plt.show()
```

### Subplots: two panels

```python
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(episodes, rewards)
ax1.set_ylabel("Reward")
ax1.set_title("Reward per episode")
ax2.plot(episodes, np.cumsum(rewards))
ax2.set_ylabel("Cumulative reward")
ax2.set_xlabel("Episode")
plt.tight_layout()
plt.show()
```

### Heatmap (e.g. value function or grid search)

```python
# 4x4 value grid
V = np.random.randn(4, 4)
plt.imshow(V, cmap="viridis")
plt.colorbar(label="V(s)")
plt.xlabel("col")
plt.ylabel("row")
plt.title("State value function")
plt.show()
```

### Saving

```python
plt.savefig("learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
```

---

## Exercises

**Exercise 1.** Plot a line of \\(y = x^2\\) for \\(x\\) in \\([0, 5]\\) with 50 points. Add labels "x" and "y", a title "y = x²", and a grid. Save the figure as `parabola.png`.

**Exercise 2.** Simulate 3 runs of 100 episodes each (random rewards, e.g. increasing mean over time). Plot all three runs on the same axes with different colors and `alpha=0.5`. Add a fourth curve that is the **mean across runs** at each episode (thicker line, no alpha). Use a legend.

**Exercise 3.** Create a 2×2 subplot layout. In the top-left, plot episode vs reward; in the top-right, episode vs episode length (dummy data); in the bottom-left, a histogram of rewards; in the bottom-right, a scatter of reward vs length. Use `fig, axes = plt.subplots(2, 2)` and index `axes[0,0]`, etc.

**Exercise 4.** Create a 5×5 matrix of values (e.g. `np.random.randn(5, 5)`). Plot it as a heatmap with `imshow` and a colorbar. Set the aspect to "equal" so cells are square. Add integer row/column labels 0–4.

**Exercise 5.** Plot two curves: (1) raw rewards per episode (noisy), (2) the same rewards smoothed with a moving average of window 10. Use `np.convolve(rewards, np.ones(10)/10, mode="valid")` for the smoothed series. Note that the smoothed series is shorter; plot it against the correct episode indices so the x-axis aligns.

**Exercise 6.** Create a 1×2 subplot: left = reward vs episode (one run); right = histogram of rewards. Use `plt.subplots(1, 2)`. **In RL:** Histograms help you see the distribution of returns across episodes.

**Exercise 7.** Plot a value function \\(V(s)\\) for a 3×3 grid: create a 3×3 array (e.g. random or from a formula), use `imshow` with `origin="lower"` so row 0 is at the bottom, and add a colorbar. Label axes "x" and "y". **In RL:** This is how you visualize tabular value functions.

**Exercise 8.** (Challenge) Simulate 5 runs of 50 episodes each (random rewards). Plot each run with low alpha; overlay the **mean ± standard error** (std/sqrt(5)) as a thicker line with a shaded band (e.g. `plt.fill_between(x, mean - se, mean + se, alpha=0.3)`). **In RL:** Standard error bands show whether differences between algorithms are significant.

---

## Professor's hints

- **In RL:** Always plot **smoothed** reward (moving average) along with raw so you can see the trend; raw alone is too noisy. Window 10–20 is typical.
- Use `plt.tight_layout()` before `show()` or `savefig()` to avoid clipped labels. For papers, use `dpi=150` or higher and `bbox_inches="tight"`.
- When comparing algorithms, use consistent colors and labels and a legend. Multiple runs: plot each run with low alpha and overlay the mean in a solid line.
- Save figures **before** `plt.show()` if you are saving in a script; on some backends, `show()` can clear the figure.

---

## Common pitfalls

- **Valid convolution shortens the array:** `np.convolve(..., mode="valid")` returns a shorter array. Use `x_smooth = np.arange(window//2, len(rewards) - window//2 + 1)` or similar so the smoothed curve aligns with the right episode indices.
- **Forgetting to close figures in loops:** If you create many figures in a loop (e.g. one per run), call `plt.close()` after saving or showing to avoid memory growth.
- **Subplot indexing:** `axes[0, 0]` is row 0, col 0. For a single row, `axes[0]` and `axes[1]` are the first and second subplots. Check with `axes.shape` if unsure.

---

**Docs:** [matplotlib.org](https://matplotlib.org/stable/contents.html). Optional: [Seaborn](https://seaborn.pydata.org/) for statistical plots.

---

## Additional exercises

### Micro-exercises

1. `import matplotlib.pyplot as plt; plt.plot([1,2,3]); plt.show()` — run this and describe what you see.
2. Add `plt.xlabel('x')`, `plt.ylabel('y')`, `plt.title('Test')` to the plot above.
3. What does `plt.figure(figsize=(8,4))` do? Try different sizes.
4. Plot a horizontal line at y=0.5 across x=[0,100] using `plt.axhline(0.5, color='red')`.
5. `plt.bar(['A','B','C'], [0.3, 0.7, 0.5]); plt.show()` — what kind of chart is this?

{{< pyrepl code="import matplotlib.pyplot as plt\n# Micro-exercise: plot a learning curve\nreturns = [1, 3, 5, 7, 8, 9, 9, 10]\nplt.figure(figsize=(6,3))\nplt.plot(returns)\nplt.xlabel('Episode')\nplt.ylabel('Return')\nplt.title('Learning Curve')\nplt.show()" height="260" >}}

### Build something small

1. Plot the function y = x² for x in [-3, 3] (use `np.linspace(-3,3,100)`).
2. Create a side-by-side comparison (2 subplots) of epsilon=0.1 vs epsilon=0 reward curves (use `plt.subplot(1,2,1)` and `plt.subplot(1,2,2)`).
3. Plot a 5×5 heatmap of random values using `plt.imshow` with `plt.colorbar()`.

### Mini-project: Bandit learning curve with multiple lines

Plot average reward over 1000 steps for ε=0, 0.01, 0.1, 0.5 on a 5-armed bandit. Each line a different epsilon. Include a legend and title.

{{< pyrepl code="import numpy as np\nimport matplotlib.pyplot as plt\nnp.random.seed(0)\n# TODO: run bandits with different epsilon values\n# and plot average reward curves\nprint('Plot multiple epsilon learning curves here')" height="220" >}}
