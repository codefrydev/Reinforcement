---
title: "Visualization & Plotting for RL"
description: "What to plot, how to read learning curves, and when to use Matplotlib vs Chart.js for RL."
date: 2026-03-10T00:00:00Z
weight: 35
draft: false
difficulty: 2
tags: ["visualization", "plotting", "learning curves", "Matplotlib", "Chart.js", "prerequisites"]
keywords: ["visualization for RL", "learning curves", "what to plot", "Matplotlib", "Chart.js"]
roadmap_icon: "trend-up"
roadmap_color: "amber"
roadmap_phase_label: "Phase 2 · Visualization"
---

This page ties together **when** and **what** to plot in reinforcement learning, **how to read** common charts, and **which tool** to use: [Matplotlib](matplotlib/) for Python scripts and notebooks, or **Chart.js** for interactive web demos and dashboards.

---

## Why visualization matters in RL

RL training is noisy: a single run can look good or bad by chance. Plots let you see **trends** (is return going up?), **variance** (how stable is learning?), and **comparisons** (which algorithm or hyperparameter is better?). Every curriculum chapter that asks you to "plot the learning curve" is training you to diagnose and communicate results.

---

## What to plot

| What | Why |
|------|-----|
| **Episode return (or cumulative reward) vs step/episode** | The main learning curve; you want it to go up and stabilize. |
| **Smoothed return** | Raw return is very noisy; use a moving average (e.g. window 10–20) to see the trend. |
| **Mean ± std over multiple runs** | One run can be lucky; report mean and standard deviation (or confidence interval) over 3–10 seeds. |
| **Value function (e.g. heatmap)** | For gridworlds or small state spaces, visualize \\(V(s)\\) or \\(Q(s,a)\\) to see what the agent has learned. |
| **Loss or TD error over time** | Helps debug training (e.g. loss should decrease or stabilize). |
| **Comparison of algorithms** | Overlay 2–3 learning curves (different colors, labels, legend) to compare sample efficiency or final performance. |

---

## How to read a learning curve

- **Going up and flattening:** Learning is working; the agent is improving and then converging. The flat level is roughly the final performance.
- **Going up then dropping:** Possible instability (e.g. too large learning rate, or catastrophic forgetting). Try smaller LR or more stable algorithm (e.g. PPO vs plain policy gradient).
- **Staying flat or random:** The agent may not be learning: check reward scale, exploration (e.g. ε-greedy), or whether the task has any learning signal.
- **High variance (zigzag):** Normal for raw episode return. Use **smoothed** curves and **multiple runs** with mean ± std so you don’t overinterpret noise.
- **Comparing two curves:** If the mean of A is above the mean of B and their shaded regions (std or CI) don’t overlap much, A is likely better; for rigor, use statistical tests or confidence intervals (e.g. [rliable](https://github.com/google-research/rliable)).

---

## When to use Matplotlib vs Chart.js

### Matplotlib (Python)

- **Use for:** Scripts, Jupyter notebooks, papers, and any workflow where you are already in Python. The curriculum and phase projects assume you plot with Matplotlib (or similar) in Python.
- **Typical use:** After training (or during), you have arrays of episode returns, steps, etc.; you call `plt.plot()`, `plt.fill_between()` for std, and `plt.savefig()` to save. See [Matplotlib](matplotlib/) for code and exercises.

### Chart.js (JavaScript, in the browser)

- **Use for:** Interactive demos on a website, dashboards that run in the browser, or when you want to embed a chart in a blog post or tutorial without running Python. Chart.js renders in the browser from data you provide (e.g. JSON).
- **Typical use:** You have a static or dynamically updated dataset (e.g. learning curve data); you pass it to Chart.js and get a line chart, bar chart, etc. No server required if data is static.

**Rule of thumb:** If you are writing Python and running experiments, use **Matplotlib**. If you are building a web page and want a chart next to the text, use **Chart.js** (or a static image exported from Matplotlib).

---

## Minimal Chart.js example (learning curve)

You can use Chart.js from a CDN in any HTML page. Below is a minimal example: a line chart of a fake learning curve (episode vs return). Copy this into an HTML file and open it in a browser.

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<canvas id="learningCurve" width="400" height="200"></canvas>
<script>
  new Chart(document.getElementById('learningCurve'), {
    type: 'line',
    data: {
      labels: [1, 20, 40, 60, 80, 100],
      datasets: [{
        label: 'Episode return (smoothed)',
        data: [20, 45, 80, 120, 160, 195],
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.3
      }]
    },
    options: {
      scales: {
        x: { title: { display: true, text: 'Episode' } },
        y: { title: { display: true, text: 'Return' } }
      }
    }
  });
</script>
```

This produces a single line: return vs episode. In a real dashboard you would replace `labels` and `data` with your own arrays (e.g. from a JSON file or an API). Chart.js supports multiple datasets (e.g. several algorithms), legends, and tooltips out of the box.

**Inline example (same idea, using the site shortcode):** Below is a small learning-curve-style chart (episode vs return) generated with the same kind of data. It shows a typical “improving then flattening” trend.

{{< chart type="line" palette="return" title="Episode return (smoothed)" labels="0, 20, 40, 60, 80, 100" data="20, 45, 80, 120, 160, 195" xLabel="Episode" yLabel="Return" >}}

---

## Debugging RL with plots

- **Learning curve not improving:** Check reward scale (not too large or small), exploration (ε or entropy), and learning rate. Plot loss or TD error to see if the network is updating.
- **Very high variance:** Run more episodes or more seeds; use a baseline (e.g. value function) to reduce variance in policy gradient.
- **Performance collapse:** Often due to too large updates (e.g. policy gradient step size). Use clipping (PPO) or smaller LR; plot policy entropy to see if the policy became deterministic too fast.

For more on interpreting results and reporting, see the curriculum chapters on hyperparameter tuning and evaluation (e.g. Volume 5, Volume 10).

---

## Links

- [Matplotlib](matplotlib/) — Full prerequisite page with line plots, heatmaps, smoothing, and exercises.
- [rliable](https://github.com/google-research/rliable) — Library for confidence intervals and aggregate metrics on RL runs.
- [Chart.js docs](https://www.chartjs.org/docs/latest/) — Official Chart.js documentation.

---

## Exercises

**Exercise 1 (Warm-up).** Run this code and describe what each parameter does: `plt.plot([1,2,3], [4,5,6], color='red', linestyle='--', linewidth=2, label='line')`. What does `label=` do?

**Exercise 2 (Coding).** Plot two learning curves (episode returns) for two agents over 100 episodes. Agent A has returns `[10, 15, 20, 25, 30, 35, 40, 45, 50, 55]` (10 samples). Agent B has `[5, 8, 12, 18, 25, 35, 40, 42, 44, 45]` (10 samples). Both on the same axes, with legend, title "Agent Comparison", xlabel "Episode", ylabel "Return".

{{< pyrepl code="import matplotlib.pyplot as plt\n# TODO: plot two learning curves\nepisodes = list(range(1, 11))\nagent_a = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]\nagent_b = [5, 8, 12, 18, 25, 35, 40, 42, 44, 45]\n# plt.plot(...)\n# Add legend, title, labels\nprint('Plot your curves!')" height="260" >}}

**Exercise 3 (Challenge).** Extend exercise 2 to add a rolling mean (window=3) for each agent. Use a simple Python loop: for each index i, `rolling_mean[i]` = mean of last 3 values.

**Exercise 4 (Variant).** Change the y-axis scale to logarithmic: `plt.yscale('log')`. When would this be useful in RL? *(Answer: when reward spans multiple orders of magnitude, e.g., from 1 to 10000.)*

**Exercise 5 (Debug).** The code below plots a learning curve but has a bug — it shows nothing. Find and fix it.

{{< pyrepl code="import matplotlib.pyplot as plt\nreturns = [5, 10, 15, 20, 25]\n# BUG: missing plt.show() or display call\nplt.figure()\nplt.plot(returns)\nplt.title('Learning Curve')\n# TODO: add what is missing\nprint('Should show a plot')" height="200" >}}

**Exercise 6 (Conceptual).** Explain the difference between a learning curve (return vs episode) and a loss curve (loss vs training step). Which one tells you if the agent is improving at the task? Which one tells you if the neural network is learning?

**Exercise 7 (RL-specific).** Why is smoothing (rolling average) important for RL learning curves? What would an unsmoothed curve look like for a stochastic environment?

**Exercise 8 (Recall).** List 3 things you should always include in any RL learning curve plot to make it reproducible and readable.
