---
title: "Chapter 63: Curiosity-Driven Exploration (ICM)"
description: "ICM: forward model, prediction error as intrinsic reward; A2C on maze."
date: 2026-03-10T00:00:00Z
weight: 63
draft: false
tags: ["ICM", "curiosity", "intrinsic reward", "A2C", "maze", "curriculum"]
keywords: ["ICM", "curiosity-driven exploration", "forward model", "intrinsic reward", "A2C"]
---

**Learning objectives**

- **Implement** the Intrinsic Curiosity Module: a forward model that predicts next-state features from current state and action.
- **Use** prediction error (between predicted and actual next features) as intrinsic reward and combine it with A2C.
- **Explain** why prediction error encourages exploration in novel or stochastic parts of the state space.
- **Compare** exploration behavior (e.g. coverage, time to goal) with and without ICM on a sparse-reward maze.
- **Relate** curiosity-driven exploration to **robot navigation** and **game AI** where rewards are sparse.

**Concept and real-world RL**

**Curiosity-driven exploration** gives the agent an intrinsic reward for visiting states where its **forward model** makes large prediction errors—i.e. where the world is surprising or novel. The Intrinsic Curiosity Module (ICM) learns a feature encoder and a forward model; the intrinsic reward is the error in predicting next features from current state and action. In **game AI** (e.g. mazes, Atari), this helps the agent explore without dense rewards; in **robot navigation**, curiosity can drive discovery of new regions before any goal reward is seen. The combination with A2C (or any policy-gradient method) balances extrinsic reward and intrinsic curiosity.

**Where you see this in practice:** ICM and similar curiosity modules in Atari and robotics; forward-model prediction error as exploration bonus in several deep RL papers.

**Illustration (prediction error as reward):** ICM uses forward model prediction error as intrinsic reward; the agent seeks states where the model is wrong. The chart below shows mean intrinsic reward over training.

{{< chart type="line" palette="return" title="Mean intrinsic reward (ICM)" labels="0, 100, 200, 300, 400" data="0.5, 0.4, 0.3, 0.25, 0.2" xLabel="Episode" yLabel="Intrinsic reward" >}}

**Exercise:** Implement the Intrinsic Curiosity Module: train a forward model to predict next state features given current state and action, and use prediction error as intrinsic reward. Combine with A2C on a sparse-reward environment (e.g., a maze).

**Professor's hints**

- Use a **feature network** (e.g. small CNN or MLP) to map raw state to a vector; the forward model takes (feature, action) and predicts next feature. Intrinsic reward = \\(\\|\\phi(s') - \\hat{\\phi}(s')\\|^2\\) (or similar).
- Scale intrinsic reward (e.g. normalize or use a coefficient) so it does not overwhelm the extrinsic reward; tune the balance.
- Start with a small deterministic maze (e.g. 10×10) so you can verify that the agent explores more and reaches the goal faster with ICM than with A2C alone.
- Optionally use an **inverse model** (predict action from \\(\\phi(s), \\phi(s')\\)) to encourage features that are actionable; the ICM paper uses both forward and inverse.

**Common pitfalls**

- **Features that ignore the agent's action:** If the forward model does not take the action as input, prediction error may be high in stochastic regions rather than "controllable" novelty; include the action.
- **Intrinsic reward too large:** If the curiosity bonus dominates, the agent may ignore the goal; scale or clip the intrinsic reward.
- **Unstable training:** The feature encoder and forward model are trained jointly with the policy; use a stable learning rate and consider freezing the feature encoder for a few steps if needed.

{{< collapse summary="Worked solution (warm-up: curiosity)" >}}
**Key idea:** Curiosity (e.g. ICM): intrinsic reward = prediction error of a forward model (predict next feature from current feature and action). The agent is curious about states where the model is wrong (surprising transitions). The feature encoder is trained so that the forward model is *not* trivial to predict (inverse model); this avoids the "noisy TV" problem where random noise gives high curiosity.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence, why does prediction error in a forward model tend to be high in states the agent has rarely visited?
2. **Coding:** Implement a minimal ICM (feature network + forward model) on a 5×5 gridworld. Log intrinsic reward per step and plot it over training. Do visited cells get lower intrinsic reward?
3. **Challenge:** Add an **inverse model** (action predictor from \\(\\phi(s), \\phi(s')\\)) and use the loss proposed in the ICM paper (forward + inverse). Compare exploration with forward-only vs full ICM.
4. **Variant:** Scale the intrinsic reward coefficient β from 0.01 to 1.0. Plot coverage and task return for each value. What happens at very high β — does the agent ignore task reward entirely?
5. **Debug:** An ICM agent shows high intrinsic reward throughout training and never converges. The forward model is a randomly initialized MLP that is never updated (its optimizer was not included in the training loop). Identify the bug and explain how a permanently random forward model creates pathological exploration behavior (the "noisy TV" problem).
6. **Conceptual:** ICM uses the inverse model to ensure features φ(s) capture only action-controllable aspects of the environment. Why is this important? Describe a scenario where raw pixel features would fail as a curiosity signal.
