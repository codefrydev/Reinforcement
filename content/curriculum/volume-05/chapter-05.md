---
title: "Chapter 45: Coding PPO from Scratch"
description: "Full PPO for LunarLanderContinuous with GAE and rollout buffer."
date: 2026-03-10T00:00:00Z
weight: 45
draft: false
---

**Learning objectives**

- Implement a **full PPO agent** for **LunarLanderContinuous-v2**: policy (actor) and value (critic) networks, rollout buffer, GAE for advantages, and multiple **epochs** of **minibatch** updates per rollout.
- **Tune** key hyperparameters (learning rate, clip \\(\epsilon\\), GAE \\(\lambda\\), batch size, number of epochs) to achieve successful landings.
- Relate each component (clip, GAE, value loss, entropy bonus) to stability and sample efficiency.

**Concept and real-world RL**

**PPO** in practice: collect a rollout of transitions (e.g. 2048 steps), compute GAE advantages, then perform several epochs of minibatch updates on the same data (policy loss with clip + value loss + entropy bonus). The **rollout buffer** stores states, actions, rewards, log-probs, and values; after each rollout we compute advantages and then iterate over minibatches. **LunarLanderContinuous** is a 2D landing task with continuous thrust; it is a standard testbed for PPO. In **robot control** and **game AI**, this "collect rollout → multiple PPO epochs" loop is the core of most on-policy algorithms.

**Where you see this in practice:** LunarLander and similar envs are used in tutorials and benchmarks; the same PPO structure scales to MuJoCo and Atari.

**Exercise:** Implement a full PPO agent for the LunarLanderContinuous-v2 environment. Use a rollout buffer, compute advantages via GAE, and perform multiple epochs of minibatch updates. Tune hyperparameters to achieve successful landing.

**Professor's hints**

- Rollout: run the policy for N steps (e.g. 2048), store (s, a, r, log_prob, V(s), done). Then compute returns and GAE from rewards and V(s). Append V(s) for the last state (or 0 if done).
- Update: for K epochs (e.g. 4–10), shuffle and split the rollout into minibatches. For each minibatch, compute ratio = π(a|s) / π_old(a|s), clipped loss, value loss (MSE to returns), entropy; total loss = -L_CLIP + c1 * value_loss - c2 * entropy. Backward and step.
- LunarLanderContinuous: state dim 8, action dim 2 (main engine, side boosters). Reward is positive for landing, negative for crashing and fuel. Success: land without crashing and get positive total return.

**Common pitfalls**

- **Reusing old log_probs:** You must store \\(\\log \\pi_{old}(a|s)\\) during rollout and use it for the ratio \\(r_t = \\pi(a|s) / \\pi_{old}(a|s)\\). Do not recompute the old policy after updating.
- **Advantage normalization:** Normalize advantages (zero mean, unit var) per rollout so the scale does not depend on return magnitude; helps with learning rate.

**Extra practice**

1. **Warm-up:** Why do we do multiple epochs of updates on the same rollout data? What is the risk if we do too many epochs?
2. **Coding:** Implement PPO for LunarLanderContinuous. Plot episode return every 10 episodes. How many episodes until you first get a successful landing (positive return)?
3. **Challenge:** Ablate: (a) remove the entropy bonus; (b) set \\(\\epsilon = 0\\) (no clip). How does learning stability and final performance change?
