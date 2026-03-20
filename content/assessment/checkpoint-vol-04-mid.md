---
title: "Checkpoint: Volume 4, Midpoint (After Chapter 35)"
description: "5 quick questions after Chapters 31–35 of Volume 4. Check you're ready to continue."
date: 2026-03-19T00:00:00Z
draft: false
tags: ["checkpoint", "volume 4", "assessment", "policy gradients", "REINFORCE", "actor-critic"]
weight: 12
roadmap_icon: "book"
roadmap_color: "indigo"
roadmap_phase_label: "Vol 4 Mid-Point"
---

Take this checkpoint after completing Chapters 31–35 (introduction to policy gradients, REINFORCE, actor-critic methods). All 5 should feel manageable — if any are unclear, re-read the relevant chapter before continuing.

---

**Q1.** Write the policy gradient theorem — the expression for ∇J(θ).

{{< collapse summary="Answer" >}}
**∇J(θ) = E_{π_θ} [ ∇_θ log π_θ(a|s) · Q^π(s,a) ]**

In words: the gradient of the expected return with respect to the policy parameters θ is the expected value of the **score function** (∇_θ log π_θ(a|s)) weighted by the **action-value** Q^π(s,a).

The score function tells us which direction to push the parameters to make action a more probable in state s; Q^π(s,a) weights that push by how good the action was.

For REINFORCE (Monte Carlo), Q^π(s,a) is replaced by the sampled return G_t.
{{< /collapse >}}

---

**Q2.** Why does REINFORCE have high variance?

{{< collapse summary="Answer" >}}
REINFORCE uses the **full Monte Carlo return G_t** as the estimate of Q^π(S_t, A_t). This return is a sum of many random rewards over the rest of the episode, and the randomness compounds across all future steps:

- Each reward is stochastic (environment noise).
- The sequence of actions taken is stochastic (policy randomness).
- G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + … accumulates all that noise.

Because we use a **single sampled trajectory** rather than an average, the gradient estimate fluctuates wildly from episode to episode. High variance → slow convergence, requiring many samples and small learning rates.
{{< /collapse >}}

---

**Q3.** What does a baseline b(s) do in a policy gradient update?

{{< collapse summary="Answer" >}}
A baseline b(s) is subtracted from the return in the policy gradient update:

∇J(θ) ≈ ∇_θ log π_θ(a|s) · (G_t − b(s))

It **reduces variance without introducing bias** (as long as b(s) depends only on s, not a).

Intuition: instead of reinforcing an action by its absolute return, we reinforce it relative to a baseline "how good is this state on average?" Actions that do better than b(s) are reinforced; actions that do worse are suppressed. Centering around zero reduces the magnitude of the gradient signal and its variance.

The most common baseline is V^π(s), leading to the advantage A(s,a) = Q(s,a) − V(s).
{{< /collapse >}}

---

**Q4.** In actor-critic, what does the critic estimate?

{{< collapse summary="Answer" >}}
The **critic** estimates the **value function** — typically V^π(s), the state-value function under the current policy π.

The critic's estimate is used to:
1. Compute the **TD error** δ = r + γ V(s') − V(s), which serves as a low-variance estimate of the advantage.
2. Provide a **baseline** b(s) = V(s) for the actor's policy gradient update.

The actor uses the critic's signal to update the policy parameters; the critic itself is trained using standard value-based methods (e.g. TD(0)). This creates a two-network architecture: one for the policy (actor), one for the value estimate (critic).
{{< /collapse >}}

---

**Q5.** Write the formula for the advantage function A(s, a).

{{< collapse summary="Answer" >}}
**A(s, a) = Q(s, a) − V(s)**

The advantage measures how much better action a is compared to the **average action** in state s under the current policy:
- **A(s,a) > 0**: action a is better than average → increase its probability.
- **A(s,a) < 0**: action a is worse than average → decrease its probability.
- **A(s,a) = 0**: action a is exactly average.

In practice, A(s,a) ≈ δ_t = R_{t+1} + γ V(S_{t+1}) − V(S_t) (the one-step TD error), which is a biased but low-variance estimate of the true advantage.
{{< /collapse >}}

---

All 5 correct? Continue to Chapter 36 (PPO and advanced policy optimization). Stuck on 2 or more? Re-read Chapters 32–34.
