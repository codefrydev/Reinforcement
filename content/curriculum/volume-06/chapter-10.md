---
title: "Chapter 60: Visualizing Model-Based Rollouts"
description: "Plot true vs predicted states; compounding error visualization."
date: 2026-03-10T00:00:00Z
weight: 60
draft: false
tags: ["model-based", "rollouts", "compounding error", "visualization", "curriculum"]
keywords: ["model rollouts", "compounding error", "predicted vs true states", "visualization"]
---

**Learning objectives**

- For a **learned dynamics model** (e.g. from Chapter 52), **sample a starting state** and generate a **rollout** of predicted states for a **fixed action sequence**.
- **Plot** the **true states** (from the environment) and the **predicted states** (from the model) on the same axes to **visualize compounding error**.
- Interpret the plot: where does the model diverge from reality?

**Concept and real-world RL**

**Visualizing** model rollouts vs real rollouts makes compounding error concrete: small 1-step errors accumulate and the predicted trajectory drifts. In **robot navigation** and **model-based RL**, this motivates short rollouts, ensemble methods, and uncertainty-aware planning. The same idea applies to **trading** models (predictions diverge over time) and **dialogue** (conversation dynamics).

**Where you see this in practice:** Debugging world models; papers show predicted vs actual trajectories.

**Illustration (compounding error):** Predicted states diverge from true states as the rollout length increases. The chart below shows MSE between predicted and true state over 10 steps.

{{< chart type="line" palette="return" title="Prediction MSE vs rollout step" labels="1, 2, 4, 6, 8, 10" data="0.02, 0.05, 0.15, 0.35, 0.6, 1.0" xLabel="Rollout step" yLabel="MSE" >}}

**Exercise:** For the learned model from Chapter 52, sample a starting state and generate a rollout of predicted states for a fixed action sequence. Plot the true states from the environment and the predicted states to visualize compounding error.

**Professor's hints**

- Fix a seed; get s0 from the env. Generate a fixed action sequence (e.g. random or sinusoidal). Run the env with these actions to get s1, s2, ... (true). Run the model: s0, a0 → ŝ1; ŝ1, a1 → ŝ2; ... (predicted).
- Plot: e.g. state dimension 0 vs time, and state dimension 1 vs time (or position vs velocity). Two curves: true and predicted. They should match early and diverge later.
- Compounding: note the time step at which the curves separate noticeably; relate to 1-step MSE.

**Common pitfalls**

- **Same actions:** Use the *same* action sequence for both env and model so the comparison is fair.
- **Terminal state:** If the episode ends in the env, stop; the model may not have a terminal prediction, so just plot up to the min of (env steps, desired horizon).

{{< collapse summary="Worked solution (warm-up: model vs model-free comparison)" >}}
**Key idea:** To compare model-based and model-free: run both for the same number of *environment steps* (e.g. 200k). Plot mean return vs steps. Model-based (e.g. MBPO, Dreamer) often reaches a given return in fewer env steps because it uses imagined data; model-free may need more steps but has no model error. Report both final return and sample efficiency (steps to reach threshold).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** What does it mean if the predicted and true curves diverge after 5 steps?
2. **Coding:** Implement the visualization for CartPole (plot 4 state dims over 20 steps). Use a model trained in Chapter 52. Save the figure and describe where error grows.
3. **Challenge:** Train the model with **more data** (50k steps) and repeat the visualization. Does the divergence delay (more steps before curves separate)?
4. **Variant:** Visualize prediction quality under two different policies: a random policy and a trained PPO policy. Does model accuracy differ between trajectories from a random vs a competent policy? Why might distribution of training data matter?
5. **Debug:** A world-model evaluation shows near-zero MSE for position but very high MSE for velocity. The model architecture has separate output heads for position and velocity, but the velocity head was initialized with 10× larger weights. Explain how this causes the issue and how to fix it.
6. **Conceptual:** How can you use prediction error over a held-out trajectory as an early warning that a MBPO or Dreamer agent is about to fail? Describe a practical monitoring scheme for a deployed model-based agent.
