---
title: "Chapter 36: Advantage Actor-Critic (A2C)"
description: "A2C for CartPole with TD error as advantage; sync multi-env."
date: 2026-03-10T00:00:00Z
weight: 36
draft: false
---

**Learning objectives**

- Implement **A2C** (Advantage Actor-Critic): actor updated with TD error as advantage, critic updated to minimize TD error.
- Use the TD error \\(r + \\gamma V(s') - V(s)\\) as the advantage (optionally with \\(V(s').detach()\\)).
- Run **multiple environments synchronously** to collect a batch of transitions and update on the batch (reduces variance further).

**Concept and real-world RL**

**A2C** is the synchronous version of A3C: the agent runs \\(N\\) environments in parallel, collects a batch of transitions, and performs one update from the batch. The advantage is the TD error (or n-step return minus V(s)). Synchronous batching makes the updates more stable than fully asynchronous A3C. In **game AI** and **robot control**, A2C is a simple and effective baseline; it is often used with a shared feature extractor (one backbone, actor and critic heads) to save parameters and improve learning.

**Where you see this in practice:** A2C is used in OpenAI Baselines and many tutorials. The same pattern (multi-env, shared backbone, TD advantage) appears in PPO implementations.

**Exercise:** Implement A2C for CartPole. Use the TD error \\(r + \\gamma V(s') - V(s)\\) as the advantage. Use a shared feature extractor or separate networks. Synchronously run multiple environments to collect data.

**Professor's hints**

- Multi-env: use `gym.vector.VectorEnv` or a list of envs; step all at once and stack states (batch dimension). Policy and V take batched input; sample one action per env.
- Shared backbone: one MLP from state to hidden features; actor head (hidden → logits) and critic head (hidden → scalar). Forward once per state batch.
- Update: collect a batch of (s, a, r, s', done). Compute \\(V(s), V(s')\\) for all; \\(\delta = r + \gamma (1-\mathrm{done}) V(s') - V(s)\\) (use `V(s').detach()`). Actor loss = mean over batch of \\(-\log \pi(a|s) \, \delta\\); critic loss = mean of \\(\delta^2\\). Backward and step.
- Hyperparameters: try 4–8 envs, batch of 32–128 steps, learning rate 1e-3 to 3e-4.

**Common pitfalls**

- **Not detaching V(s') in δ for actor:** If you backprop \\(\delta\\) through \\(V(s')\\), the actor gradient will try to change the critic’s prediction for \\(s'\\), which is not the intended objective.
- **Batch dimension mismatch:** Ensure state batch has shape (batch_size, state_dim); action and \\(\delta\\) have shape (batch_size,) or (batch_size, 1) for gathering log-probs.

**Extra practice**

1. **Warm-up:** What is the difference between A2C and REINFORCE in terms of what signal is used to update the policy (return \\(G_t\\) vs TD error \\(\delta\\))? Which has higher variance?
2. **Coding:** Implement A2C with 4 parallel CartPole envs. Plot mean episode return (over the 4 envs) every 10 updates. Compare training time (wall clock) to single-env REINFORCE for the same number of steps.
3. **Challenge:** Use **n-step returns** (e.g. n=5): collect 5 steps per env, compute \\(G_{t:t+5} = \sum_{i=0}^{4} \gamma^i r_{t+i} + \gamma^5 V(s_{t+5})\\), and use \\(G_{t:t+5} - V(s_t)\\) as advantage. Compare stability and sample efficiency to 1-step TD.
