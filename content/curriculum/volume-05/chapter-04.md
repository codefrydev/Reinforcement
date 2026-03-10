---
title: "Chapter 44: PPO: Implementation Details"
description: "Generalized Advantage Estimation (GAE) function."
date: 2026-03-10T00:00:00Z
weight: 44
draft: false
---

**Learning objectives**

- Implement **Generalized Advantage Estimation (GAE)**: compute advantage estimates \\(\\hat{A}_t\\) from a trajectory of rewards and value estimates using \\(\\gamma\\) and \\(\\lambda\\).
- Write the recurrence: \\(\\hat{A}_t = \\delta_t + (\\gamma\\lambda) \\delta_{t+1} + (\\gamma\\lambda)^2 \\delta_{t+2} + \\cdots\\) where \\(\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)\\).
- Use GAE in a PPO (or actor-critic) pipeline so advantages are fed into the policy loss.

**Concept and real-world RL**

**GAE** (Generalized Advantage Estimation) provides a bias–variance trade-off for the advantage: \\(\\hat{A}_t^{GAE} = \\sum_{l=0}^{\\infty} (\\gamma\\lambda)^l \\delta_{t+l}\\). When \\(\\lambda=0\\), \\(\\hat{A}_t = \\delta_t\\) (1-step TD, low variance, high bias). When \\(\\lambda=1\\), \\(\\hat{A}_t = G_t - V(s_t)\\) (Monte Carlo, high variance, low bias). Tuning \\(\\lambda\\) (e.g. 0.95–0.99) balances the two. In **robot control** and **game AI**, GAE is the standard way to compute advantages for PPO and actor-critic; it is implemented with a backward loop over the trajectory.

**Where you see this in practice:** GAE is used in almost every PPO and A2C implementation (OpenAI Baselines, Stable-Baselines3, CleanRL).

**Exercise:** Implement Generalized Advantage Estimation (GAE) for a trajectory. Write a function that takes rewards and value estimates and returns GAE advantages for each timestep using \\(\lambda\\) and \\(\gamma\\).

**Professor's hints**

- Inputs: rewards \\(r_0, \\ldots, r_{T-1}\\), values \\(V(s_0), \\ldots, V(s_T)\\) (or one less for last state), \\(\\gamma\\), \\(\\lambda\\). Don't forget the bootstrap \\(V(s_T)\\) for the last step (or use 0 if terminal).
- Recursion: \\(\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)\\). Then \\(\\hat{A}_t = \\delta_t + \\gamma\\lambda \\hat{A}_{t+1}\\). Implement by looping backward from \\(t = T-1\\) to 0.
- Output: array of advantages of length T (one per step). Optionally normalize advantages to zero mean and unit variance before feeding to PPO.

**Common pitfalls**

- **Off-by-one in value indices:** Ensure \\(V(s_{t+1})\\) is used in \\(\\delta_t\\); for the last step, use \\(V(s_T)\\) if non-terminal or 0 if done.
- **Done flag:** If the episode ends at step \\(T\\), set \\(V(s_T)=0\\) (or mask the bootstrap). Otherwise you will use the value of a state that starts the next episode.

**Extra practice**

1. **Warm-up:** For \\(\\lambda=0\\), what is \\(\\hat{A}_t\\) in terms of \\(\\delta_t\\)? For \\(\\lambda=1\\), what is \\(\\hat{A}_t\\) in terms of \\(G_t\\) and \\(V(s_t)\\)?
2. **Coding:** Implement `gae(rewards, values, gamma=0.99, lambda_=0.95)` where `values[i]` is \\(V(s_i)\\) and length is len(rewards)+1. Test on a short trajectory of length 5; check that for \\(\\lambda=0\\) you get \\(\\hat{A}_t = \\delta_t\\).
3. **Challenge:** Vectorize the backward loop (e.g. with NumPy or PyTorch) so you can compute GAE for a batch of trajectories at once.
