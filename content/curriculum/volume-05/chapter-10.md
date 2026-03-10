---
title: "Chapter 50: Advanced Hyperparameter Tuning"
description: "Weights & Biases sweep for SAC on custom env."
date: 2026-03-10T00:00:00Z
weight: 50
draft: false
tags: ["hyperparameter tuning", "Weights and Biases", "SAC", "sweep", "curriculum"]
keywords: ["hyperparameter tuning", "Weights and Biases", "wandb sweep", "SAC"]
---

**Learning objectives**

- Use **Weights & Biases** (or similar) to run a **hyperparameter sweep** for SAC on your custom environment (or a standard one).
- Sweep over **learning rate**, **entropy coefficient** (or auto-\\(\\alpha\\) target), and **network size** (hidden dims).
- **Visualize** the effect on **final return** and **learning speed** (e.g. steps to reach a threshold).

**Concept and real-world RL**

Hyperparameter tuning is essential for getting the best from RL algorithms; **sweeps** (grid or random search over learning rate, network size, etc.) are standard in research and industry. **Weights & Biases** (wandb) logs metrics and supports sweep configs; similar tools include MLflow, Optuna, and Ray Tune. In **robot control** and **game AI**, tuning learning rate and entropy (or clip range for PPO) often has the largest impact. Automating sweeps saves time and makes results reproducible.

**Where you see this in practice:** Papers and codebases report sweep ranges; W&B and Optuna are common in RL projects.

**Illustration (hyperparameter sweep):** Different learning rates yield different final returns. The chart below shows mean final return (over 3 seeds) for 4 learning rate values.

{{< chart type="bar" title="Final return vs learning rate (3 seeds)" labels="1e-4, 3e-4, 1e-3, 3e-3" data="4200, 4800, 4500, 3200" >}}

**Exercise:** Use Weights & Biases to sweep over learning rates, entropy coefficient, and network sizes for SAC on your custom environment. Visualize the effect on final return and learning speed.

**Professor's hints**

- Define a sweep config (YAML or dict): e.g. method "grid" or "random", metric "eval/mean_return", parameters lr (log uniform 1e-4 to 1e-2), network_size (values [64, 128, 256]), etc. Run multiple agents (one per config).
- Log to wandb: `wandb.init(project="sac-sweep", config=config)`; log `wandb.log({"return": mean_return}, step=step)`. Use wandb sweep to launch runs.
- Visualize: parallel coordinates plot (each run = line, color = return); or scatter (lr vs final return). Identify which lr and network size work best.

**Common pitfalls**

- **Too few runs per config:** Run at least 2–3 seeds per config so you see variance; otherwise one lucky seed can mislead.
- **Sweep too large:** Start with 2–3 key hyperparameters (lr, entropy/alpha, hidden size); add more only if needed.

{{< collapse summary="Worked solution (warm-up: hyperparameter tuning)" >}}
**Key idea:** For tuning, fix the rest and vary one (or two) key hyperparameters: e.g. learning rate, \\(\\alpha\\) or entropy coefficient, clip range for PPO, or network size. Run a few seeds per setting and compare mean final return. Use a small grid first (e.g. 3 values); expand only if the best is at the boundary. Document the best setting and the metric (e.g. mean return over last 100 episodes).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why is it important to run multiple seeds when comparing hyperparameters?
2. **Coding:** Run a small grid: lr in [1e-4, 3e-4], hidden size in [64, 256]. For each of the 4 configs, run 2 seeds for 100k steps. Report mean and std of final return. Which config wins?
3. **Challenge:** Use **Bayesian optimization** (e.g. Optuna or wandb sweep with bayes) to suggest the next hyperparameters given past results. Compare with random search after 20 runs.
