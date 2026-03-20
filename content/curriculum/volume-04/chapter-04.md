---
title: "Chapter 34: Reducing Variance in Policy Gradients"
description: "State-value baseline with REINFORCE; compare gradient variance."
date: 2026-03-10T00:00:00Z
weight: 34
draft: false
difficulty: 7
tags: ["REINFORCE", "baseline", "variance reduction", "curriculum"]
keywords: ["variance reduction", "baseline", "state-value baseline", "policy gradient"]
roadmap_color: "amber"
roadmap_icon: "trend-up"
roadmap_phase_label: "Vol 4 · Ch 4"
---

**Learning objectives**

- Add a **state-value baseline** \\(V(s)\\) to REINFORCE and explain why it reduces variance without introducing bias (when the baseline does not depend on the action).
- Train the baseline network (e.g. MSE to fit returns \\(G_t\\)) alongside the policy.
- Compare the **variance of gradient estimates** (e.g. magnitude of parameter updates or variance of \\(G_t - b(s_t)\\)) with and without baseline.

**Concept and real-world RL**

The policy gradient with a baseline is \\(\mathbb{E}[ \nabla \log \pi(a|s) \, (G_t - b(s)) ]\\). If \\(b(s)\\) does not depend on the action \\(a\\), this is still an unbiased estimate of \\(\nabla J\\); the baseline only changes the variance. A natural choice is \\(b(s) = V^\pi(s)\\), the expected return from state \\(s\\). Then the term \\(G_t - V(s_t)\\) is an estimate of the **advantage** (how much better this trajectory was than average). In **game AI** or **robot control**, lower-variance gradients mean faster and more stable learning; baselines are standard in actor-critic and PPO.

**Where you see this in practice:** Value baselines are used in REINFORCE with baseline, A2C, A3C, PPO, and most policy gradient implementations. The idea generalizes to advantage estimators (GAE, TD error).

**Illustration (variance reduction):** With a baseline, the variance of the gradient estimate typically decreases as \\(V(s)\\) improves. The chart below compares the magnitude of updates (with vs without baseline) over training.

{{< chart type="bar" palette="comparison" title="Gradient estimate magnitude (last 100 steps)" labels="Without baseline, With baseline" data="2.5, 0.8" yLabel="Magnitude" >}}

**Exercise:** Add a state-value baseline to your REINFORCE implementation. Train the baseline network alongside the policy. Compare the variance of gradient estimates (e.g., by tracking the magnitude of updates) with and without baseline.

**Professor's hints**

- Baseline network: same state input as policy, output scalar \\(V(s)\\). Loss for baseline = \\(\sum_t (G_t - V(s_t))^2\\) (MSE). Train it on the same trajectories (e.g. after each episode, update both policy and baseline).
- Policy gradient term: use \\((G_t - V(s_t))\\) instead of \\(G_t\\) in the REINFORCE loss. Use `V(s_t).detach()` so gradients do not flow through the baseline into the policy (or flow them if you want a shared representation).
- Variance comparison: log \\(\| \nabla_\theta \|^2\\) (squared norm of gradient) or the variance of \\(G_t - b(s_t)\\) over a batch. With a good baseline, both should be smaller.

**Common pitfalls**

- **Baseline that depends on action:** If \\(b\\) depends on \\(a\\), the gradient estimate can become biased. The baseline must be a function of state (and optionally time) only.
- **Training baseline too aggressively:** If the baseline fits \\(G_t\\) too well, \\(G_t - V(s_t)\\) becomes near zero and the policy gradient signal vanishes. Use a separate network or a slower learning rate for the baseline so it tracks but does not overfit.

{{< collapse summary="Worked solution (warm-up: why baseline keeps gradient unbiased)" >}}
**Warm-up:** \\(\\mathbb{E}[ \\nabla \\log \\pi(a|s) \\cdot b(s) ] = \\sum_a \\pi(a|s) \\nabla \\log \\pi(a|s) \\cdot b(s) = b(s) \\sum_a \\nabla \\pi(a|s) = b(s) \\nabla (\\sum_a \\pi(a|s)) = b(s) \\nabla 1 = 0\\). So subtracting \\(b(s)\\) does not change the expectation of the gradient; it only reduces variance when \\(b(s)\\) approximates \\(\\mathbb{E}[G_t|s]\\). That’s why we use \\(G_t - V(s)\\) or \\(G_t - b(s)\\) in REINFORCE with baseline.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence, why does subtracting a state-dependent baseline \\(b(s)\\) from \\(G_t\\) keep the policy gradient unbiased? (Hint: \\(\mathbb{E}[ \nabla \log \pi \, b(s) ] = ?\\))
2. **Coding:** Implement REINFORCE with baseline. Plot the mean and standard deviation of \\(G_t - V(s_t)\\) over the last 100 steps (or one episode) every 500 episodes. Does the std decrease as V gets better?
3. **Challenge:** Use a **moving average** baseline \\(b(s) = \bar{G}\\) (global average return) instead of a learned V(s). Implement it and compare variance and learning speed to the learned baseline.
4. **Variant:** Try using a **constant** baseline (e.g. the mean return from the previous episode) versus a learned \\(V(s)\\). Does the constant baseline reduce variance? Which performs better after 500 episodes?
5. **Debug:** The code below uses a baseline that depends on the action \\(a_t\\) (specifically, it uses \\(Q(s,a)\\) as the baseline), introducing bias. Explain why this is wrong and how to fix it.

{{< pyrepl code="def biased_policy_gradient(log_probs, states, actions, returns, Q_table):\n    \"\"\"BUG: baseline depends on the action, introducing bias.\"\"\"\n    loss = 0\n    for lp, s, a, G in zip(log_probs, states, actions, returns):\n        # BUG: Q(s,a) depends on a, so E[grad log pi * Q(s,a)] != 0\n        baseline = Q_table.get((s, a), 0.0)\n        loss += -lp * (G - baseline)\n    return loss\n\n# Fix: use V(s) = E_a[Q(s,a)] as baseline, NOT Q(s,a)\n# E[grad log pi * V(s)] = 0 since V(s) does not depend on a\nprint('Fix: baseline must not depend on the current action')" height="200" >}}

6. **Conceptual:** Explain mathematically why \\(\\mathbb{E}[\\nabla_\\theta \\log \\pi(a|s) \\cdot b(s)] = 0\\) when \\(b\\) depends only on the state \\(s\\) (not on \\(a\\)).
7. **Recall:** State the advantage definition \\(A^\\pi(s,a) = Q^\\pi(s,a) - V^\\pi(s)\\) from memory and explain why \\(G_t - V(s_t)\\) estimates it.
