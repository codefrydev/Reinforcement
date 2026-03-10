---
title: "Prerequisites — Tools & Libraries"
description: "Learn the stack used in this curriculum: Python, NumPy, Pandas, Matplotlib, PyTorch, TensorFlow, and more."
date: 2026-03-10T00:00:00Z
draft: false
---

Learn or brush up on the tools and libraries used across the [curriculum](/curriculum/) and in the [preliminary assessment](/preliminary/). Each topic page includes detailed explanations, multiple examples, and **5 exercises** to practice before tackling the chapter exercises.

**Why RL needs each:** Python builds trajectories, configs, and agent classes. NumPy handles state/observation arrays and batch math. Matplotlib plots learning curves and value heatmaps. PyTorch/TensorFlow implement Q-networks and policy gradients with autograd. Gym gives standard envs (reset, step) used in every exercise.

**Recommended order:** Python → NumPy → Pandas → Matplotlib → PyTorch → TensorFlow → Gym (optional) → Other libraries.

**One small task per topic (check you're ready):**

| Topic | Task |
|-------|------|
| [Python](/prerequisites/python/) | Write a function that takes a list of (state, action, reward) tuples and returns the list of rewards. |
| [NumPy](/prerequisites/numpy/) | Compute the discounted return \\(r_0 + \\gamma r_1 + \\gamma^2 r_2\\) for `rewards = np.array([0, 0, 1])` and `gamma = 0.9` using NumPy (no Python loop). |
| [Matplotlib](/prerequisites/matplotlib/) | Plot a simple learning curve: x = episode index (1..100), y = cumulative reward per episode (use random or fake data). |
| [PyTorch](/prerequisites/pytorch/) | Create a tensor `x` with `requires_grad=True`, compute `y = x**2`, call `y.backward()`, and print `x.grad` for `x=2.0`. |
| [Gym](/prerequisites/gym/) | Run 10 steps of CartPole with random actions; print the total reward and the final observation. |

When you can do these, take the **[Phase 2 readiness quiz](/assessment/phase-2-readiness/)**.
