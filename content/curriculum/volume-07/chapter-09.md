---
title: "Chapter 69: RL² (Reinforcement Learning as an RNN)"
description: "RNN policy with (state, action, reward, done) input; POMDP tasks."
date: 2026-03-10T00:00:00Z
weight: 69
draft: false
---

**Learning objectives**

- **Implement** RL²: an RNN policy whose input at each step is (state, action, reward, done) from the previous step (and current state), and whose output is the action.
- **Explain** how the RNN hidden state can encode the "learning algorithm" or belief about the task from the history of experience.
- **Train** the RNN on multiple POMDP tasks (or tasks with different dynamics/rewards) so that it learns to adapt its behavior from the history.
- **Evaluate** the trained policy on new POMDP tasks and compare with a non-recurrent policy.
- **Relate** RL² to **dialogue** (context-dependent response) and **game AI** (adapting to different levels or opponents).

**Concept and real-world RL**

**RL² (Reinforcement Learning as an RNN)** treats the learning algorithm itself as a recurrent policy: the agent receives (state, action, reward, done) as input (plus current state) and outputs the next action. The RNN's **hidden state** accumulates information from the trajectory, effectively implementing an adaptive strategy that can change with experience. This is useful in **POMDPs** where the true state is hidden and the agent must infer the task or state from history, and in **multi-task** settings where the task is not explicitly given. In **dialogue** and **game AI**, the "task" or context is only revealed through interaction, so an RNN that conditions on history is a natural fit.

**Where you see this in practice:** RL² and similar "learning to learn" RNN policies; meta-RL with recurrent policies.

**Exercise:** Implement RL²: train an RNN policy that takes as input (state, action, reward, done) and outputs actions. The RNN's hidden state should encode the learning algorithm. Train on multiple POMDP tasks.

**Professor's hints**

- **Input:** At step \\(t\\), feed (s_t, a_{t-1}, r_{t-1}, done_{t-1}) and optionally s_{t-1}. For \\(t=0\\), use dummy values (e.g. 0, 0, 0, 0) or a special token. The RNN outputs \\(a_t\\) (e.g. logits for a discrete action).
- **POMDP tasks:** Use environments where the state is partially observed (e.g. only last K observations, or a noisy state), or use different tasks (e.g. different goal locations) so the agent must use history to infer the task. Train on a distribution of such tasks.
- **Training:** Use policy gradient (e.g. REINFORCE or PPO) on the full trajectory; the RNN is unrolled over the episode. Backprop through time over the full episode length.
- Start with short episodes (e.g. 20–50 steps) and a small RNN (e.g. 1–2 layer LSTM) to debug.

**Common pitfalls**

- **Forgetting to pass previous (a, r, done):** The key of RL² is that the policy conditions on the full history via (s, a, r, done); if you only pass state, it reduces to a standard policy and won't adapt.
- **Credit assignment:** Long episodes make BPTT difficult; use truncated BPTT or a policy gradient that doesn't require backprop through the whole episode (e.g. REINFORCE with baseline).
- **Task distribution:** If all tasks are too similar, the RNN may not learn meaningful adaptation; ensure tasks differ (e.g. different goals, different transition dynamics) so that history is informative.

**Extra practice**

1. **Warm-up:** Why might an RNN that receives (state, action, reward, done) be able to "adapt" to a new task without explicit gradient updates?
2. **Coding:** Implement a minimal RL²: 2D gridworld with 3 different goal positions; each episode sample one goal (unknown to the policy). Input = (position, prev_action, prev_reward, done). Train with REINFORCE on 1000 episodes. Test on held-out goal positions.
3. **Challenge:** Use a **bandit or simple POMDP** where the reward distribution of each arm (or state) is unknown and different per task. Train RL² to learn to explore and exploit from the history of (state, action, reward). Compare with a standard bandit algorithm that gets the task explicitly.
