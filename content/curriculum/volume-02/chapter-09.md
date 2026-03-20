---
title: "Chapter 19: Hyperparameter Tuning in Tabular RL"
description: "Grid search over α and ε for Q-learning on Cliff Walking."
date: 2026-03-10T00:00:00Z
weight: 19
draft: false
difficulty: 6
tags: ["hyperparameter tuning", "Q-learning", "grid search", "Cliff Walking", "curriculum"]
keywords: ["hyperparameter tuning", "grid search", "alpha epsilon", "Q-learning"]
roadmap_color: "teal"
roadmap_icon: "book"
roadmap_phase_label: "Vol 2 · Ch 9"
---

**Learning objectives**

- Run a grid search over learning rate \\(\alpha\\) and exploration \\(\epsilon\\) for Q-learning.
- Aggregate results over multiple trials (e.g. mean reward per episode) and visualize with a heatmap.
- Interpret which hyperparameter combinations work best and why.

**Concept and real-world RL**

**Hyperparameters** (e.g. \\(\alpha\\), \\(\epsilon\\), \\(\gamma\\)) strongly affect learning speed and final performance. **Grid search** tries every combination in a predefined set; it is simple but costly when there are many parameters. In practice, RL tuning often uses grid search for 2–3 key parameters, or Bayesian optimization / bandit-based tuning for larger spaces. Reporting mean and std over multiple seeds is essential because RL is noisy. Heatmaps (e.g. \\(\alpha\\) vs \\(\epsilon\\) with color = mean reward) make good and bad regions visible at a glance.

**Illustration (grid search results):** For different \\(\alpha\\) values (with \\(\epsilon\\) fixed), mean episode return after training can vary. The chart below shows typical mean return for \\(\alpha \in \\{0.1, 0.5, 0.9\\}\\) on Cliff Walking (mean over last 100 episodes, 10 seeds).

{{< chart type="bar" palette="comparison" title="Mean return vs α (ε=0.1, 10 seeds)" labels="α=0.1, α=0.5, α=0.9" data="-14, -12, -18" yLabel="Mean return" >}}

**Exercise:** For Q-learning on the Cliff Walking, perform a grid search over \\(\alpha \in \\{0.1, 0.5, 0.9\\}\\) and \\(\epsilon \in \\{0.01, 0.1, 0.3\\}\\). For each combination, run 10 independent trials and report mean cumulative reward per episode. Visualize the results as a heatmap.

**Professor's hints**

- Nested loop: for each \\((\alpha, \\epsilon)\\), run 10 trials (different random seeds). For each trial, run Q-learning for a fixed number of episodes (e.g. 500) and record the *mean* reward per episode over that run (or the mean of the last 100 episodes). Then average the 10 trial means for that \\((\alpha, \\epsilon)\\).
- Heatmap: rows = \\(\\epsilon\\), columns = \\(\\alpha\\) (or vice versa). Color = mean cumulative reward (or mean of last N episodes). Use `plt.imshow` or `seaborn.heatmap`. Label axes with the actual \\(\\alpha\\) and \\(\\epsilon\\) values.
- Typical outcome: very low \\(\\epsilon\\) may not explore enough; very high \\(\\alpha\\) may be unstable. Mid-range values often do best. Report the best \\((\alpha, \\epsilon)\\) and the worst.

**Common pitfalls**

- **Too few trials:** One trial per combination is noisy. Use at least 5–10 seeds and report mean (and optionally std) so you can see variance.
- **Wrong metric:** Use a metric that reflects final performance (e.g. mean reward over last 100 episodes, or total reward in a fixed episode count). Do not use only the first 10 episodes.
- **Same seed for all:** Use different seeds per trial (e.g. seed = trial_id or random). Otherwise "10 trials" are identical and you get no variance estimate.

{{< collapse summary="Worked solution (warm-up: mean and std over 3 trials)" >}}
**Warm-up:** For one (\\(\\alpha\\), \\(\\epsilon\\)) pair, run 3 trials and compute mean and standard deviation of mean reward per episode. Why is std useful? **Answer:** Mean = (R1 + R2 + R3)/3; std = sqrt of variance of those 3 numbers. Std is useful because RL is noisy—different seeds give different results. Reporting mean ± std (or std error) shows whether a hyperparameter choice is reliably good or just lucky in one run. In papers we often report mean ± std over 5–10 seeds.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For one (\\(\\alpha\\), \\(\\epsilon\\)) pair, run 3 trials and compute the mean and standard deviation of the mean reward per episode. Why is std useful?
2. **Coding:** Run a small hyperparameter grid: α in [0.01, 0.1], ε in [0.05, 0.2] on Cliff Walking (or gridworld). For each of the 4 combinations, run 3 seeds and record mean return. Plot a bar chart or table of mean ± std.
3. **Challenge:** Add \\(\\gamma\\) to the grid (e.g. 0.9, 0.99). You now have 3 parameters; use a small grid (e.g. 2×2×2) or fix one parameter and show 2D heatmaps for two values of \\(\\gamma\\).
4. **Variant:** Run a finer grid: \\(\alpha \\in \\{0.05, 0.1, 0.2, 0.5\\}\\) with \\(\epsilon\\) fixed at 0.1. Is performance sensitive to small changes in \\(\alpha\\)? Is there a clear best value?
5. **Debug:** The code below uses the same random seed for all trials, making all 3 trials identical. Fix it so each trial uses a different seed.

{{< pyrepl code="import random\n\ndef run_trial(alpha, eps, seed=42):  # BUG: same seed always\n    random.seed(seed)\n    returns = []\n    for ep in range(100):\n        ep_return = random.gauss(-13, 3)  # simulated\n        returns.append(ep_return)\n    return sum(returns) / len(returns)\n\nresults = [run_trial(0.1, 0.1) for _ in range(3)]  # identical!\nprint('Results (should differ):', results)\n\n# Fix: pass different seeds, e.g. seed=trial_id or seed=None\nresults_fixed = [run_trial(0.1, 0.1, seed=i) for i in range(3)]\nprint('Fixed results:', results_fixed)" height="240" >}}

6. **Conceptual:** Why do we report mean ± standard deviation over multiple seeds rather than just the best run?
7. **Recall:** Name one advantage of Bayesian optimization over grid search for hyperparameter tuning.
