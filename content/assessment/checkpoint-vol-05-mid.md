---
title: "Checkpoint: Volume 5, Midpoint (After Chapter 45)"
description: "5 quick questions after Chapters 41–45 of Volume 5. Check you're ready to continue."
date: 2026-03-19T00:00:00Z
draft: false
tags: ["checkpoint", "volume 5", "assessment", "PPO", "TRPO", "GAE", "SAC"]
---

Take this checkpoint after completing Chapters 41–45 (TRPO, PPO, and GAE). All 5 should feel manageable — if any are unclear, re-read the relevant chapter before continuing.

---

**Q1.** What does PPO's clipping mechanism prevent?

{{< collapse summary="Answer" >}}
PPO's clipping prevents **excessively large policy updates** that would move the new policy too far from the old policy in a single gradient step.

Without clipping, if a batch of experience produces a very large advantage for some action, the gradient update could change the policy drastically — potentially "forgetting" what was learned or collapsing to a degenerate policy.

The clip keeps the probability ratio r(θ) = π_θ(a|s) / π_{θ_old}(a|s) within [1−ε, 1+ε] (typically ε = 0.2), ensuring each update is a small, stable improvement. This achieves a similar goal to TRPO's trust region constraint, but with a simple first-order method instead of a constrained optimization.
{{< /collapse >}}

---

**Q2.** Write the PPO clipped objective — what is the ratio r(θ)?

{{< collapse summary="Answer" >}}
**r(θ) = π_θ(a|s) / π_{θ_old}(a|s)**

This is the **probability ratio** between the new policy and the old policy (the policy that collected the data).

The PPO clipped objective is:

L^CLIP(θ) = E_t [ min( r_t(θ) Â_t ,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]

Where Â_t is the advantage estimate. The min of the unclipped and clipped objectives ensures:
- When the advantage is positive, we increase the action's probability, but not beyond 1+ε.
- When the advantage is negative, we decrease the action's probability, but not below 1−ε.
- The min prevents any "gaming" of the objective by taking larger steps than the clipping allows.
{{< /collapse >}}

---

**Q3.** What does GAE (Generalized Advantage Estimation) interpolate between?

{{< collapse summary="Answer" >}}
GAE interpolates between the **one-step TD error** (high bias, low variance) and the **full Monte Carlo return** (low bias, high variance) via the parameter λ ∈ [0, 1]:

**Â_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}**

Where δ_{t+l} = R_{t+l+1} + γ V(S_{t+l+1}) − V(S_{t+l}) are one-step TD errors.

- **λ = 0**: reduces to the one-step TD advantage δ_t = r + γV(s') − V(s). Low variance, high bias.
- **λ = 1**: reduces to the full Monte Carlo advantage G_t − V(s). Low bias, high variance.
- **Intermediate λ**: a geometric blend, trading off bias and variance. λ = 0.95 is common in practice.
{{< /collapse >}}

---

**Q4.** PPO is on-policy. What does that mean for data reuse?

{{< collapse summary="Answer" >}}
Being on-policy means PPO **must discard data after each policy update**.

After collecting a batch of experience with policy π_{θ_old}, PPO can perform several gradient update steps (typically 3–10 epochs) on that same batch — but only while the policy stays close to π_{θ_old} (enforced by clipping). Once the policy parameters θ have been updated, the old data is **stale**: it was collected under π_{θ_old}, not the current π_θ.

Consequences:
- **Sample inefficient** compared to off-policy methods like SAC or TD3 that reuse millions of transitions from a replay buffer.
- PPO must **continuously collect new data** from the environment to continue training.
- This makes PPO more suitable for fast simulators (where data is cheap) and less suitable for real-world robotics (where data is expensive).
{{< /collapse >}}

---

**Q5.** When would you prefer SAC over PPO?

{{< collapse summary="Answer" >}}
Prefer **SAC (Soft Actor-Critic)** over PPO when:

1. **Data is expensive**: SAC is off-policy and uses a replay buffer, reusing every transition many times. This makes it far more sample-efficient than PPO.
2. **Continuous action spaces**: SAC was designed for continuous control (e.g. robotics, locomotion). It maximizes a reward + entropy objective, producing a naturally exploratory, stochastic policy.
3. **Exploration is hard**: The entropy regularization in SAC encourages broad exploration, preventing premature convergence to suboptimal policies.
4. **Stable hyperparameters matter**: SAC automatically adjusts the entropy temperature, reducing the need for hand-tuning.

Prefer **PPO** when:
- Data is cheap (fast simulator, many parallel environments).
- The action space is discrete.
- Simplicity and robustness to hyperparameters are priorities.
- You need a well-understood baseline for research comparisons.
{{< /collapse >}}

---

All 5 correct? Continue to Chapter 46 (SAC and off-policy actor-critic methods). Stuck on 2 or more? Re-read Chapters 42–44.
