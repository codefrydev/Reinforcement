---
title: "Phase 4: Deep RL — Milestones & Coding Challenges"
description: "Checkpoints after Vol 3–5; DQN and PPO coding challenges."
date: 2026-03-10T00:00:00Z
draft: false
---

Phase 4 is [Volume 3: Value Function Approximation & Deep Q-Learning](../../curriculum/volume-03/), [Volume 4: Policy Gradients](../../curriculum/volume-04/), and [Volume 5: Advanced Policy Optimization](../../curriculum/volume-05/) (chapters 21–50). Use the milestones below, then try the coding challenges and take the Deep RL quiz.

---

## Milestone checkpoints

- **After Volume 3 (DQN and variants):** You can run **DQN** on CartPole (or similar): replay buffer, target network, ε-greedy. You understand why function approximation is needed and how the TD target is computed.
- **After Volume 4 (Policy gradients):** You can implement **REINFORCE** or **A2C** on CartPole. You understand the policy gradient theorem and the role of a baseline (value function).
- **After Volume 5 (PPO, SAC):** You can run **PPO** on LunarLander or CartPole and (optionally) **SAC** on a continuous control task. You understand the clipped objective (PPO) or max-entropy (SAC).

---

## Coding challenges

1. **DQN on CartPole:** Tune hyperparameters (learning rate, replay size, target update frequency, ε schedule) so that your DQN reaches an average episode return of **195 or more** over 100 episodes in **under 50,000 env steps**. Plot the learning curve (mean return vs steps). Document the hyperparameters you used.

2. **Double DQN vs DQN:** Implement Double DQN (use the online network to select the action, target network to evaluate it). Compare with standard DQN on CartPole: plot both learning curves and the **mean Q-value** (over a fixed set of states or over the batch) over training. Does Double DQN show less overestimation (lower or more stable Q-values)?

3. **REINFORCE vs PPO:** Implement REINFORCE (with or without baseline) and PPO on CartPole. Compare **sample efficiency**: how many env steps does each need to reach 195 average return? Plot learning curves. Discuss why PPO is often more sample-efficient and stable.

When you are done, take the **[Phase 4 Deep RL quiz](../../assessment/phase-4-deep-rl/)**.
