---
title: "Chapter 33: The REINFORCE Algorithm"
description: "REINFORCE for CartPole with softmax policy; note variance."
date: 2026-03-10T00:00:00Z
weight: 33
draft: false
tags: ["REINFORCE", "CartPole", "softmax policy", "variance", "curriculum"]
keywords: ["REINFORCE algorithm", "policy gradient", "CartPole", "variance"]
---

**Learning objectives**

- Implement **REINFORCE** (Monte Carlo policy gradient): estimate \\(\nabla_\theta J\\) using the return \\(G_t\\) from full episodes.
- Use a neural network policy with **softmax** output for discrete actions (e.g. CartPole).
- Observe and explain the **high variance** of gradient estimates when using raw returns \\(G_t\\) (no baseline).

**Concept and real-world RL**

**REINFORCE** is the simplest policy gradient algorithm: run an episode under \\(\pi_\theta\\), compute the return \\(G_t\\) from each step, and update \\(\theta\\) with \\(\theta \leftarrow \theta + \alpha \sum_t G_t \nabla_\theta \log \pi(a_t|s_t)\\). It is on-policy and Monte Carlo (needs full episodes). The variance of \\(G_t\\) can be large, especially in long episodes, which makes learning slow or unstable. In **game AI**, REINFORCE is a baseline for more advanced methods (actor-critic, PPO); in **robot control**, it is rarely used alone because of sample efficiency and variance. Adding a **baseline** (e.g. state-value function) reduces variance without introducing bias.

**Where you see this in practice:** REINFORCE appears in early deep RL (e.g. policy gradient for Atari); modern practice usually replaces it with actor-critic or PPO. It is still used in bandits and simple episodic tasks for teaching.

**Illustration (REINFORCE learning curve):** Episode returns often have high variance early on. The chart below shows a typical trend: noisy at first, then improving and eventually stabilizing.

{{< chart type="line" palette="return" title="Episode return (REINFORCE on CartPole)" labels="0, 100, 200, 300, 400, 500" data="25, 80, 120, 150, 180, 195" xLabel="Episode" yLabel="Return" >}}

**Exercise:** Implement REINFORCE for CartPole. Use a neural network policy that outputs action probabilities via softmax. Train with baseline (optional) and plot the episode returns. Note the high variance.

**Professor's hints**

- Policy network: input state (4-dim for CartPole), hidden layer(s), output 2 logits; apply softmax to get \\(\pi(a|s)\\). Sample action, compute \\(\log \pi(a_t|s_t)\\), store states, actions, rewards for the episode.
- After the episode, compute returns \\(G_t\\) from each step (loop backward or use discount factor). Loss = \\(-\sum_t G_t \log \pi(a_t|s_t)\\) (negative because we minimize loss but want to maximize return). Then `loss.backward()` and optimizer step.
- Optional baseline: subtract a state-dependent baseline \\(b(s_t)\\) from \\(G_t\\) in the loss (e.g. a learned V(s)); estimate V with another network or running average.
- Plot episode return over time; you will likely see high variance (returns jump up and down). Compare with and without baseline if implemented.

**Common pitfalls**

- **Using reward instead of return:** The gradient uses \\(G_t\\) (return from step \\(t\\) to end), not just \\(r_t\\). Sum discounted rewards from \\(t\\) onward.
- **Wrong sign on loss:** PyTorch minimizes loss; we want to maximize return, so loss = -\\( \sum_t G_t \log \pi(a_t|s_t) \\) (minus the policy gradient objective).
- **Not normalizing returns (optional):** For stability, some implementations normalize \\(G_t\\) to zero mean and unit variance over the batch; helps with learning rate.

{{< collapse summary="Worked solution (warm-up: G_t and REINFORCE update)" >}}
**Warm-up:** \\(G_0 = 0 + 0.9\\cdot 0 + 0.81\\cdot 1 = 0.81\\), \\(G_1 = 0 + 0.9\\cdot 1 = 0.9\\), \\(G_2 = 1\\). REINFORCE update (one term per step): \\(\\theta \\leftarrow \\theta + \\alpha \\sum_t \\nabla_\\theta \\log \\pi(a_t|s_t) \\cdot G_t\\). So we multiply the log-probability gradient at each step by the return from that step; higher returns get more weight, which pushes the policy toward actions that led to high return.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For a 3-step episode with rewards [0, 0, 1] and \\(\gamma=0.9\\), write \\(G_0, G_1, G_2\\). Then write the REINFORCE update (one term per step) in symbols.
2. **Coding:** Implement REINFORCE without a baseline for CartPole. Log the standard deviation of the last 10 episode returns every 100 episodes. Does it decrease over training?
3. **Challenge:** Add a simple baseline \\(b(s) = V_\phi(s)\\) (one hidden layer). Train V to minimize \\((G_t - V(s_t))^2\\) and use \\(G_t - V(s_t)\\) in the policy gradient. Compare variance of updates and learning speed with and without baseline.
4. **Variant:** Run REINFORCE with \\(\gamma=0.99\\) and \\(\gamma=0.9\\) on CartPole. Does discounting the future aggressively change what the policy learns to optimize? Which reaches 195 average return faster?
5. **Debug:** The code below uses the per-step reward \\(r_t\\) instead of the return \\(G_t\\) when weighting the log-probability gradient — a very common mistake. Fix it.

{{< pyrepl code="def reinforce_loss_buggy(log_probs, rewards, gamma=0.9):\n    \"\"\"BUG: uses r_t instead of G_t in the update.\"\"\"\n    loss = 0\n    for t, (lp, r) in enumerate(zip(log_probs, rewards)):\n        loss += -lp * r  # BUG: should use G_t, not r_t\n    return loss\n\ndef compute_returns(rewards, gamma=0.9):\n    G, returns = 0, []\n    for r in reversed(rewards):\n        G = r + gamma * G\n        returns.insert(0, G)\n    return returns\n\n# Demo with a 3-step episode\nrewards = [0, 0, 1]\nreturns = compute_returns(rewards)\nprint('Returns G_t:', [round(g,3) for g in returns])\n# Fix: replace 'r' with 'G' from compute_returns" height="240" >}}

6. **Conceptual:** Why does REINFORCE have high variance? What is the key quantity that determines the scale of the gradient update, and why does it vary so much across episodes?
7. **Recall:** Write the REINFORCE weight update \\(\\theta \\leftarrow \\theta + \\alpha \\sum_t G_t \\nabla_\\theta \\log \\pi(a_t|s_t;\\theta)\\) from memory.
