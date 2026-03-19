---
title: "Chapter 41: The Problem with Standard Policy Gradients"
description: "Large step size and policy collapse in bandit; visualize probabilities."
date: 2026-03-10T00:00:00Z
weight: 41
draft: false
tags: ["policy gradient", "policy collapse", "step size", "bandit", "curriculum"]
keywords: ["policy gradient problems", "policy collapse", "step size", "bandit"]
---

**Learning objectives**

- Demonstrate how a **too-large step size** in policy gradient updates can cause **policy collapse** (e.g. one action gets probability near 1 too quickly) and loss of exploration.
- **Visualize** policy probabilities over time in a simple bandit problem under different learning rates.
- Relate this to the motivation for **trust region** and **clipped** methods (e.g. PPO, TRPO).

**Concept and real-world RL**

Standard policy gradient \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\) can be unstable: a single bad batch or a large step can make the policy assign near-zero probability to previously good actions (**policy collapse**). In a **multi-armed bandit** (or a simple MDP), this is easy to see: with a large \\(\alpha\\), the policy can become deterministic too fast and get stuck. In **robot control** and **game AI**, we want to avoid catastrophic updates; **PPO** (clipped objective) and **TRPO** (KL constraint) limit how much the policy can change per update. This chapter illustrates the problem in a minimal setting.

**Where you see this in practice:** PPO and TRPO are widely used in robotics and RL benchmarks precisely because they avoid the instability of vanilla policy gradients.

**Illustration (policy collapse):** With too large a step size, policy probabilities can collapse (e.g. one action gets probability 1). The chart below shows a healthy policy vs a collapsed one over 3 actions.

{{< chart type="bar" title="π(a|s) — healthy vs collapsed" labels="Action 1, Action 2, Action 3" data="0.4, 0.35, 0.25" >}}

**Exercise:** In a simple bandit problem, implement a policy gradient update with too large a step size. Show how it can lead to a collapse in performance. Visualize the policy probabilities over time.

**Professor's hints**

- Bandit: K arms, policy \\(\pi(a) = \mathrm{softmax}(\\theta)\\). Sample action, get reward, update \\(\theta\\) with REINFORCE: \\(\theta_a \leftarrow \theta_a + \\alpha \, (r - b) \, (1 - \\pi(a))\\) for the chosen action (and similar for others in the gradient). Use a large \\(\alpha\\) (e.g. 0.5 or 1.0) and watch \\(\pi\\) become near-one-hot quickly; then try a smaller \\(\alpha\\).
- Visualize: plot \\(\pi(a)\\) for each arm over iterations. With large \\(\alpha\\), one arm dominates fast; with small \\(\alpha\\), probabilities change smoothly.
- Performance collapse: if the best arm has high variance, a few bad samples can push the policy away from it; with large steps, recovery is slow or impossible.

**Common pitfalls**

- **Baseline:** Use a baseline (e.g. running average of rewards) so the update is not purely driven by raw reward magnitude; the collapse effect is still visible with large \\(\alpha\\).
- **Initialization:** Start with uniform or near-uniform \\(\theta\\) so you can see the policy move; if you start already peaked, the effect is less clear.

{{< collapse summary="Worked solution (warm-up: why large step size collapses policy)" >}}
**Warm-up:** A very large step size can push the softmax probabilities so far toward one action that the policy becomes nearly deterministic; once one action dominates, its gradient gets most of the updates and the others get little signal, so the policy "collapses" to that action. Smaller \\(\\alpha\\) or a baseline/KL penalty keeps updates moderate and avoids this. This is why PPO and TRPO use constrained or clipped updates.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence, why does a very large policy gradient step size risk "collapsing" the policy to one action?
2. **Coding:** Implement the bandit policy gradient with \\(\alpha \\in \\{0.01, 0.1, 0.5\\}\\). For each, plot policy probabilities over 500 steps. Which one collapses fastest?
3. **Challenge:** Add a **KL penalty** to the update (e.g. penalize \\(D_{KL}(\\pi_{old} \\| \\pi_{new})\\)). Does a moderate penalty prevent collapse even with larger \\(\alpha\\)?
4. **Variant:** Repeat the bandit experiment with 10 arms instead of 3. Does collapse happen faster with more arms? Which arm tends to dominate, and does it depend on the random seed?

{{< pyrepl code="import math, random\nrandom.seed(0)\n\ndef softmax(theta):\n    e = [math.exp(t) for t in theta]\n    s = sum(e)\n    return [ei/s for ei in e]\n\nk = 3  # arms\ntrue_rewards = [0.1, 0.5, -0.2]\ntheta = [0.0] * k\nalpha = 0.3  # try 0.01, 0.1, 0.3\n\nfor step in range(200):\n    pi = softmax(theta)\n    a = random.choices(range(k), weights=pi)[0]\n    r = random.gauss(true_rewards[a], 0.5)\n    # REINFORCE update (no baseline)\n    for i in range(k):\n        theta[i] += alpha * r * ((1 if i == a else 0) - pi[i])\n    if step % 50 == 0:\n        print(f'step {step}: pi={[round(p,3) for p in softmax(theta)]}')" height="260" >}}

5. **Debug:** The update below applies the gradient to all \\(\\theta_i\\) with \\(r\\) (the raw reward) rather than \\(r - b\\) (reward minus baseline), making the update very noisy when rewards have a large mean. Add a running-average baseline to fix it.

```python
# BUG: no baseline — large mean reward inflates all updates
for i in range(k):
    theta[i] += alpha * reward * ((1 if i == a else 0) - pi[i])
# Fix: replace reward with (reward - running_avg_reward)
```

6. **Conceptual:** Explain why PPO's clipping mechanism solves the collapse problem shown in this chapter. What does it actually prevent in terms of probability ratios?
7. **Recall:** Define "policy collapse" in one sentence and state one algorithmic fix (from TRPO or PPO) that prevents it.
