---
title: "Chapter 12: Temporal Difference (TD) Learning"
description: "TD(0) prediction for blackjack; compare with Monte Carlo."
date: 2026-03-10T00:00:00Z
weight: 12
draft: false
---

**Learning objectives**

- Implement TD(0) prediction: update \\(V(s)\\) using the TD target \\(r + \\gamma V(s')\\) immediately after each transition.
- Compare TD(0) with Monte Carlo in terms of convergence speed and sample efficiency.
- Understand bootstrapping: TD uses current estimates instead of waiting for episode end.

**Concept and real-world RL**

**Temporal Difference (TD) learning** updates value estimates using the **TD target** \\(r + \\gamma V(s')\\): \\(V(s) \\leftarrow V(s) + \\alpha [r + \\gamma V(s') - V(s)]\\). Unlike Monte Carlo, TD does not need to wait for the episode to end; it **bootstraps** on the current estimate of \\(V(s')\\). TD(0) often converges faster per sample and works in continuing tasks. In practice, TD is the basis for SARSA, Q-learning, and many deep RL algorithms (e.g. DQN uses a TD-like target). Blackjack lets you compare TD(0) and MC on the same policy and state space.

**Exercise:** Implement TD(0) prediction for the same blackjack policy. Compare the convergence speed and final value estimates with Monte Carlo. Use a step size \\(\alpha=0.01\\) and run for 10,000 episodes.

**Professor's hints**

- Initialize \\(V(s)=0\\) for all states (or a default). For each transition \\((s, a, r, s')\\), update \\(V(s) \\leftarrow V(s) + \\alpha [r + \\gamma V(s') - V(s)]\\). Use the *updated* \\(V\\) for the next step (online update).
- You need to run episodes and, at each step, have \\(s\\), \\(r\\), \\(s'\\). The policy is fixed (stick on 20/21, else hit). Same blackjack env as Chapter 11.
- Comparison: run MC for 10k episodes and TD for 10k episodes. Plot \\(V(s)\\) for a few states over episodes (e.g. log scale or every N episodes). TD often stabilizes faster; final estimates may be slightly different because TD is biased (bootstrapping) while MC is unbiased.

**Common pitfalls**

- **Step size:** \\(\alpha=0.01\\) is small; values change slowly. If you use a large \\(\alpha\\), TD can be noisy or unstable. For tabular blackjack, 0.01–0.1 is typical.
- **Comparing fairly:** Use the same number of episodes (10k) for both. MC needs full episodes; TD updates every step. So TD makes many more updates per episode—keep that in mind when comparing "convergence speed."
- **Initial V(s'):** For terminal \\(s'\\), \\(V(s')=0\\) by definition. Do not update \\(V\\) for terminal states; use 0 when computing the target.

**Extra practice**

1. **Warm-up:** Write the TD(0) update in one line of Python (pseudo-code): given \\(s, r, s', V, \\alpha, \\gamma\\), what is the new \\(V(s)\\)?
2. **Coding:** Implement TD(0) prediction for a fixed policy on a small tabular MDP (e.g. 4 states). Run for 1000 episodes and plot V(s) for one state over time.
3. **Challenge:** Run TD(0) with \\(\alpha \\in \\{0.001, 0.01, 0.1\\}\\) for 10k episodes. Plot the learning curve (e.g. mean absolute error from a reference \\(V\\) if you have one, or \\(V\\) for one state over time). Which \\(\\alpha\\) converges fastest? Which is most stable?
