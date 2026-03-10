---
title: "Chapter 48: SAC vs. PPO"
description: "Compare SAC and PPO on Hopper, Walker2d; when to choose which."
date: 2026-03-10T00:00:00Z
weight: 48
draft: false
tags: ["SAC", "PPO", "Hopper", "Walker2d", "comparison", "curriculum"]
keywords: ["SAC vs PPO", "Hopper", "Walker2d", "when to use SAC or PPO"]
---

**Learning objectives**

- Run **SAC** and **PPO** on the same continuous control tasks (e.g. Hopper, Walker2d).
- Compare **final performance**, **sample efficiency** (return vs env steps), and **wall-clock time**.
- Discuss when to choose one over the other (sample efficiency, stability, tuning, off-policy vs on-policy).

**Concept and real-world RL**

**SAC** is off-policy (replay buffer) and maximizes entropy; **PPO** is on-policy (rollouts) and uses a clipped objective. SAC often achieves higher sample efficiency (fewer env steps to reach good performance) but can be sensitive to hyperparameters and replay buffer size; PPO is more robust and easier to tune in many settings. In **robot control** benchmarks (Hopper, Walker2d, HalfCheetah), both are standard; in **game AI** and **RLHF**, PPO is more common. Choice depends on data cost (can we afford many env steps?), need for off-policy (e.g. using logged data), and engineering preference.

**Where you see this in practice:** Benchmarks (e.g. MuJoCo) report both; industry often standardizes on PPO for simplicity or SAC for sample efficiency.

**Illustration (SAC vs PPO sample efficiency):** For the same task, SAC often reaches a given return in fewer env steps. The chart below compares mean return vs env steps (conceptual).

{{< chart type="line" title="Mean return vs env steps (Hopper)" labels="0, 200k, 400k, 600k, 800k" data="0, 800, 1800, 2500, 3000" >}}

**Exercise:** Run both SAC and PPO on the same set of continuous control tasks (e.g., Hopper, Walker2d). Compare final performance, sample efficiency, and wall-clock time. Discuss when you might choose one over the other.

**Professor's hints**

- Same seeds and run length (e.g. 1M steps) for fair comparison. Plot mean return (over last 10 eval episodes) vs steps and vs wall-clock time.
- Sample efficiency: which algorithm reaches a given return (e.g. 2000 for Hopper) in fewer steps? Wall-clock: which is faster per step (PPO often does more compute per step due to multiple epochs)?
- When to choose: PPO when you want simplicity and stability; SAC when sample efficiency matters and you can tune (or use defaults).

**Common pitfalls**

- **Different observation/action preprocessing:** Use the same env wrapper and normalization for both so the comparison is fair.
- **Single run:** Run multiple seeds (e.g. 3–5) and report mean and std of final return.

{{< collapse summary="Worked solution (warm-up: reporting results)" >}}
**Key idea:** Always run multiple seeds (e.g. 3–5 or more) and report mean ± std (or standard error) of episode return or success rate. A single run can be lucky or unlucky; mean and std show whether an algorithm or hyperparameter is reliably good. Plot learning curves with a shaded band (mean ± std) so readers can see variance. This is standard in RL papers and benchmarks.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** List one advantage of PPO over SAC and one advantage of SAC over PPO.
2. **Coding:** Run SAC and PPO on Hopper for 500k steps each (3 seeds). Plot learning curves with standard error. Which has higher final return on average?
3. **Challenge:** On a task where PPO is sample-inefficient, try **PPO with a larger rollout** (e.g. 4096 steps) and more epochs. Does it close the gap with SAC?
