---
title: "Chapter 34: Reducing Variance in Policy Gradients"
description: "State-value baseline with REINFORCE; compare gradient variance."
date: 2026-03-10T00:00:00Z
weight: 34
draft: false
---

**Learning objectives**

- Add a **state-value baseline** \\(V(s)\\) to REINFORCE and explain why it reduces variance without introducing bias (when the baseline does not depend on the action).
- Train the baseline network (e.g. MSE to fit returns \\(G_t\\)) alongside the policy.
- Compare the **variance of gradient estimates** (e.g. magnitude of parameter updates or variance of \\(G_t - b(s_t)\\)) with and without baseline.

**Concept and real-world RL**

The policy gradient with a baseline is \\(\mathbb{E}[ \nabla \log \pi(a|s) \, (G_t - b(s)) ]\\). If \\(b(s)\\) does not depend on the action \\(a\\), this is still an unbiased estimate of \\(\nabla J\\); the baseline only changes the variance. A natural choice is \\(b(s) = V^\pi(s)\\), the expected return from state \\(s\\). Then the term \\(G_t - V(s_t)\\) is an estimate of the **advantage** (how much better this trajectory was than average). In **game AI** or **robot control**, lower-variance gradients mean faster and more stable learning; baselines are standard in actor-critic and PPO.

**Where you see this in practice:** Value baselines are used in REINFORCE with baseline, A2C, A3C, PPO, and most policy gradient implementations. The idea generalizes to advantage estimators (GAE, TD error).

**Exercise:** Add a state-value baseline to your REINFORCE implementation. Train the baseline network alongside the policy. Compare the variance of gradient estimates (e.g., by tracking the magnitude of updates) with and without baseline.

**Professor's hints**

- Baseline network: same state input as policy, output scalar \\(V(s)\\). Loss for baseline = \\(\sum_t (G_t - V(s_t))^2\\) (MSE). Train it on the same trajectories (e.g. after each episode, update both policy and baseline).
- Policy gradient term: use \\((G_t - V(s_t))\\) instead of \\(G_t\\) in the REINFORCE loss. Use `V(s_t).detach()` so gradients do not flow through the baseline into the policy (or flow them if you want a shared representation).
- Variance comparison: log \\(\| \nabla_\theta \|^2\\) (squared norm of gradient) or the variance of \\(G_t - b(s_t)\\) over a batch. With a good baseline, both should be smaller.

**Common pitfalls**

- **Baseline that depends on action:** If \\(b\\) depends on \\(a\\), the gradient estimate can become biased. The baseline must be a function of state (and optionally time) only.
- **Training baseline too aggressively:** If the baseline fits \\(G_t\\) too well, \\(G_t - V(s_t)\\) becomes near zero and the policy gradient signal vanishes. Use a separate network or a slower learning rate for the baseline so it tracks but does not overfit.

**Extra practice**

1. **Warm-up:** In one sentence, why does subtracting a state-dependent baseline \\(b(s)\\) from \\(G_t\\) keep the policy gradient unbiased? (Hint: \\(\mathbb{E}[ \nabla \log \pi \, b(s) ] = ?\\))
2. **Coding:** Implement REINFORCE with baseline. Plot the mean and standard deviation of \\(G_t - V(s_t)\\) over the last 100 steps (or one episode) every 500 episodes. Does the std decrease as V gets better?
3. **Challenge:** Use a **moving average** baseline \\(b(s) = \bar{G}\\) (global average return) instead of a learned V(s). Implement it and compare variance and learning speed to the learned baseline.
