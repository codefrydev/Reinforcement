---
title: "Chapter 98: Evaluating RL Agents"
description: "PPO on 10 seeds; mean, std; rliable confidence intervals."
date: 2026-03-10T00:00:00Z
weight: 98
draft: false
---

**Learning objectives**

- **Train** a PPO agent on **10 different random seeds** and collect **final returns** (or mean return over the last N episodes) for each seed.
- **Compute** the **mean** and **standard deviation** of these returns and report them (e.g. "mean ± std").
- **Compute** **stratified confidence intervals** (e.g. using the **rliable** library or similar) so that intervals account for both within-run and across-run variance.
- **Interpret** the results: what does the interval tell us about the agent's performance and reliability? Why is reporting only mean ± std over seeds often insufficient?
- **Relate** evaluation practice to **robot navigation**, **healthcare**, and **trading** where reliable performance estimates matter.

**Concept and real-world RL**

**Evaluating RL agents** requires multiple **runs** (seeds) because training is stochastic; a single run can be lucky or unlucky. Reporting **mean ± std** over seeds is common but can be misleading: the std captures **between-seed** variance, while **within-seed** variance (e.g. across evaluation episodes) also matters. **Stratified** or **rliable**-style confidence intervals use both sources of variance to produce intervals that have correct coverage (e.g. 95% CI that contains the true mean performance with probability 0.95). In **robot navigation**, **healthcare**, and **trading**, we need reliable estimates before deployment.

**Where you see this in practice:** rliable library; evaluation in RL papers (multiple seeds, confidence intervals); reporting standards for RL benchmarks.

**Exercise:** Train a PPO agent on 10 different random seeds. Compute the mean and standard deviation of final returns. Then compute stratified confidence intervals using the rliable library. Interpret the results.

**Professor's hints**

- **Setup:** Same env (e.g. CartPole or MuJoCo), same PPO hyperparameters; only the random seed changes (for env, policy init, and any sampling). Train each seed until "done" (e.g. 500k steps or 1M).
- **Final return:** For each seed, take the mean return over the last 50 (or 100) evaluation episodes (no exploration). So you have 10 numbers: one per seed.
- **Mean and std:** mean(R), std(R) over the 10 seeds. Report "mean ± std" and optionally "median ± MAD."
- **rliable:** Install rliable; use their functions to compute confidence intervals (e.g. stratified bootstrap or interval_overlapping_runs). Input: matrix of returns (runs × evaluation_episodes) or (runs × 1) if you only have one number per run. Output: interval and point estimate. Interpret: "We are 95% confident that the true mean return lies in [L, U]."
- **Interpretation:** Briefly explain why we need multiple seeds and why a confidence interval is more informative than mean ± std (e.g. std underestimates uncertainty when we have few seeds).

**Common pitfalls**

- **Too few seeds:** 10 is a minimum; for papers, 5–10 seeds are common but more is better. With 3 seeds, intervals are very wide.
- **Evaluation episodes:** Use a fixed number of eval episodes per seed (e.g. 50) and no exploration (deterministic policy or ε=0) so returns are comparable.
- **rliable API:** Check the library docs for the exact function (e.g. get_interval or aggregate_metrics); the input format may be a matrix (runs × episodes).

**Extra practice**

1. **Warm-up:** Why might "mean ± std over 5 seeds" be misleading when each seed's return is the mean of 50 evaluation episodes?
2. **Coding:** Train PPO on CartPole for 5 seeds, 200k steps each. For each seed, record mean return over last 50 eval episodes. Compute mean ± std. Then use rliable (or bootstrap by seed) to get a 95% CI. Report both. How much wider is the CI than mean ± std?
3. **Challenge:** Compare **interval estimation** across algorithms: train PPO and SAC (or DQN) each on 10 seeds. Compute 95% CIs for both. Do the intervals overlap? What can you conclude about which algorithm is better?
