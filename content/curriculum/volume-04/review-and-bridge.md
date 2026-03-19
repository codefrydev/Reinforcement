---
title: "Volume 4 Review & Bridge to Volume 5"
description: "Review Volume 4 (Policy Gradients, Actor-Critic, DDPG, TD3) and preview Volume 5 (PPO, TRPO, SAC — stable, scalable policy optimization)."
date: 2026-03-19T00:00:00Z
draft: false
weight: 100
tags: ["review", "bridge", "Volume 4", "Volume 5", "REINFORCE", "actor-critic", "PPO"]
---

## Volume 4 Recap Quiz (5 questions)

{{< collapse summary="Q1. Write the REINFORCE gradient estimator. What is its key weakness?" >}}
∇J(θ) ≈ (1/N) Σ_τ Σ_t ∇log π(a_t|s_t; θ) · G_t

where G_t is the Monte Carlo return from step t. **Key weakness: high variance.** G_t accumulates reward noise over the entire episode, making gradient estimates noisy — learning is slow and unstable without a baseline.
{{< /collapse >}}

{{< collapse summary="Q2. What is the advantage function A(s,a), and how does it reduce variance?" >}}
A(s,a) = Q(s,a) − V(s). It measures how much better action a is compared to the average action in state s. Using A instead of raw returns **centers the signal around zero** — good actions get positive gradient, bad actions get negative gradient, and the common "tide" (V(s)) is subtracted out. This reduces variance without introducing bias if V is exact.
{{< /collapse >}}

{{< collapse summary="Q3. In actor-critic, what do the actor and critic each do?" >}}
- **Actor**: the policy π(a|s; θ) — selects actions and is updated via policy gradient.
- **Critic**: estimates V(s; w) or Q(s,a; w) — provides the advantage/baseline signal to reduce variance in the actor's gradient.

The critic uses TD learning (bootstrapping), so the actor no longer needs to wait for full episode returns. This enables **online** (step-by-step) updates.
{{< /collapse >}}

{{< collapse summary="Q4. How does DDPG extend actor-critic to continuous actions?" >}}
DDPG (Deep Deterministic Policy Gradient) uses a **deterministic** policy μ(s; θ) that outputs a single action (not a distribution). The gradient becomes: ∇J ≈ ∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s; θ). This avoids sampling from a distribution. It also uses experience replay and target networks (from DQN) to stabilize training. TD3 extends this with twin critics and delayed policy updates.
{{< /collapse >}}

{{< collapse summary="Q5. What is the main practical problem with vanilla REINFORCE/actor-critic?" >}}
**Large policy updates destabilize training.** A bad gradient step can move the policy far into a poor region — and because the new policy collects different data, recovery is slow or impossible. This motivates Volume 5: PPO/TRPO explicitly constrain how much the policy can change per update, giving much more stable training at scale.
{{< /collapse >}}

---

## What Changes in Volume 5

| | Volume 4 (Basic Policy Gradient) | Volume 5 (Stable Policy Optimization) |
|---|---|---|
| **Update constraint** | None — step size chosen by hand | PPO: ratio clip; TRPO: KL constraint |
| **Variance reduction** | Baseline / advantage | GAE (λ-weighted advantage) |
| **Off-policy support** | Limited (DDPG/TD3) | SAC: maximum entropy, off-policy |
| **Sample efficiency** | Low (on-policy, discard after update) | Moderate (PPO epochs; SAC replay) |
| **Entropy** | Not explicit | SAC maximises entropy for exploration |

**The big insight:** Controlling the size of policy updates via clipping (PPO) or trust-region constraints (TRPO) makes training dramatically more stable. GAE smoothly interpolates between TD (low variance, bias) and MC (high variance, unbiased).

---

## Bridge Exercise: REINFORCE Variance on a Bandit

{{< pyrepl code="import random\nimport math\n\nrandom.seed(0)\n\n# 3-armed bandit: true means\ntrue_means = [0.5, 1.0, -0.5]\n\ndef softmax_policy(logits):\n    \"\"\"Convert logits to probabilities.\"\"\"\n    e = [math.exp(l) for l in logits]\n    s = sum(e)\n    return [ei/s for ei in e]\n\ndef sample_action(probs):\n    r = random.random()\n    cum = 0.0\n    for i, p in enumerate(probs):\n        cum += p\n        if r < cum:\n            return i\n    return len(probs) - 1\n\n# REINFORCE: collect returns, compute gradient\nlogits = [0.0, 0.0, 0.0]  # initial policy: uniform\nreturns_log = []\n\nfor episode in range(300):\n    probs = softmax_policy(logits)\n    action = sample_action(probs)\n    reward = random.gauss(true_means[action], 1.0)  # noisy reward\n    returns_log.append(reward)\n\n    # Policy gradient update (no baseline)\n    lr = 0.1\n    for a in range(3):\n        indicator = 1.0 if a == action else 0.0\n        logits[a] += lr * reward * (indicator - probs[a])\n\nfinal_probs = softmax_policy(logits)\nprint(f'Final policy: {[round(p,3) for p in final_probs]}')\nprint(f'Best arm is arm 1 (mean=1.0) -- prob assigned: {final_probs[1]:.3f}')\n\n# Show variance of returns\nimport statistics\nprint(f'Return std over episodes: {statistics.stdev(returns_log):.3f}  (high!)')\nprint('Baseline subtracts avg return -> centers signal -> lower variance')" height="340" >}}

**Next:** [Volume 5: PPO, TRPO & SAC](../volume-05/)
