---
title: "Chapter 52: Learning World Models"
description: "Train NN to predict next state from CartPole; compounding error."
date: 2026-03-10T00:00:00Z
weight: 52
draft: false
tags: ["world model", "dynamics model", "CartPole", "compounding error", "curriculum"]
keywords: ["learning world models", "dynamics prediction", "compounding error", "CartPole"]
---

**Learning objectives**

- Collect **random trajectories** from CartPole and train a **neural network** to predict the next state given (state, action).
- Evaluate **prediction accuracy** over 1 step, 5 steps, and 10 steps; observe **compounding error** as the horizon grows.
- Relate model error to the limitations of long-horizon model-based rollouts.

**Concept and real-world RL**

A **world model** (or dynamics model) predicts \\(s_{t+1}\\) from \\(s_t, a_t\\). We can train it on collected data (e.g. MSE loss). Errors **compound** over multi-step rollouts: a small 1-step error becomes large after many steps. In **robot navigation**, learned models are used for short-horizon planning; in **game AI** (e.g. Dreamer), models are used in latent space to reduce dimensionality and control rollouts. Understanding compounding error is key to designing model-based algorithms.

**Where you see this in practice:** World models appear in Dreamer, MBPO, and PILCO; robotics often uses short model rollouts.

**Illustration (prediction error):** A learned dynamics model's prediction error typically grows with the number of steps (compounding). The chart below shows MSE between predicted and true next state over rollout length.

{{< chart type="line" palette="return" title="State prediction MSE vs rollout step" labels="1, 2, 3, 4, 5, 6" data="0.01, 0.03, 0.08, 0.18, 0.35, 0.6" xLabel="Rollout step" yLabel="MSE" >}}

**Exercise:** Collect random trajectories from the CartPole environment. Train a neural network to predict the next state given current state and action. Evaluate its prediction accuracy over multiple steps and note the compounding error.

**Professor's hints**

- Data: run random policy for 10k steps, store (s, a, s'). Train: input (s, a), target s'; use MSE. Split train/val.
- Evaluation: from a held-out s0, rollout the model for 1, 5, 10 steps (feeding predicted s as input). Compare predicted s_t to true s_t from the env (same actions). Plot MSE vs step; it typically grows.
- Compounding: small errors at each step accumulate; the model never saw its own predictions during training (distribution shift).

**Common pitfalls**

- **Training on same distribution as evaluation:** When you rollout the model, you feed it *predicted* states, not real ones. So long-horizon error reflects distribution shift (model sees its own errors).
- **Normalization:** Normalize state and action for training; denormalize for env comparison.

{{< collapse summary="Worked solution (warm-up: world model)" >}}
**Key idea:** Train a model \\(\\hat{s}_{t+1} = f(s_t, a_t)\\) (and optionally \\(\\hat{r}_t\\)) on collected transitions. Use it to generate imagined rollouts for planning or for training a policy in the latent space. Evaluate by 1-step and multi-step prediction error (MSE). The model improves sample efficiency when it is accurate; long-horizon rollout error typically grows (compounding).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why does 10-step prediction error usually exceed 1-step error?
2. **Coding:** Train the model; plot 1-step, 5-step, 10-step MSE on a validation set. Fit a curve (e.g. exponential) to error vs step.
3. **Challenge:** Use an **ensemble** of 5 models and take the mean prediction. Does ensemble reduce compounding error over 10 steps?
4. **Variant:** Change the ensemble size from 5 to 2 and to 10. Plot multi-step MSE for each ensemble size. Is there diminishing return beyond a certain size?
5. **Debug:** The following forward model is trained but multi-step error explodes after 3 steps even though 1-step error is low. The model predicts `next_state = f(state, action)` but the training loop feeds `next_state_predicted` back as input during training. What is wrong, and how do you fix it?
6. **Conceptual:** Explain why compounding model error is analogous to floating-point rounding error accumulating over repeated multiplication. What structural property of the model would slow this accumulation?
