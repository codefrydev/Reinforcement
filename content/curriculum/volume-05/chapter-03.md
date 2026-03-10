---
title: "Chapter 43: Proximal Policy Optimization (PPO): Intuition"
description: "Clipped surrogate objective; contrast with unclipped."
date: 2026-03-10T00:00:00Z
weight: 43
draft: false
tags: ["PPO", "clipped surrogate", "proximal policy", "curriculum"]
keywords: ["PPO", "Proximal Policy Optimization", "clipped objective", "surrogate"]
---

**Learning objectives**

- Explain in your own words how the **clipped surrogate objective** in PPO prevents too-large policy updates without solving a constrained optimization (unlike TRPO).
- Write the **clipped loss** \\(L^{CLIP}(\\theta)\\) and the **unclipped** (ratio-based) objective; contrast when they differ.
- Relate the **clip range** \\(\epsilon\\) (e.g. 0.2) to how much the policy can change in one update.

**Concept and real-world RL**

**PPO** (Proximal Policy Optimization) keeps the policy update conservative by **clipping** the probability ratio \\(r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}\\). The objective is \\(L^{CLIP} = \\mathbb{E}[ \\min( r_t \\hat{A}_t, \\mathrm{clip}(r_t, 1-\\epsilon, 1+\\epsilon) \\hat{A}_t ) ]\\): if the advantage is positive, we do not let the ratio exceed \\(1+\\epsilon\\); if negative, we do not let it go below \\(1-\\epsilon\\). So we never encourage a huge increase in probability for a good action (which could overshoot) or a huge decrease for a bad one. In **robot control**, **game AI**, and **dialogue**, PPO is the default choice for policy gradient because it is simple, stable, and effective.

**Where you see this in practice:** PPO is used in OpenAI Five, robotics, RLHF, and most continuous control benchmarks.

**Illustration (clipped objective):** The PPO clip prevents the probability ratio from moving too far from 1. The chart below shows the ratio \\(\\pi(a|s)/\\pi_{old}(a|s)\\) before and after clipping (conceptual).

{{< chart type="line" palette="return" title="Probability ratio (before/after clip)" labels="Step 1, Step 2, Step 3, Step 4" data="1, 1.1, 1.15, 1.2" xLabel="Step" yLabel="Ratio" >}}

**Exercise:** Explain in your own words how the clipped surrogate objective in PPO prevents too large policy updates. Write the clipped loss function and contrast it with the unclipped version.

**Professor's hints**

- Unclipped: \\(L = \\mathbb{E}[ r_t \\hat{A}_t ]\\). If \\(r_t\\) is large (policy much more likely to take \\(a_t\\) now), the gradient can be huge when \\(\\hat{A}_t > 0\\).
- Clipped: replace \\(r_t \\hat{A}_t\\) with \\(\\min(r_t \\hat{A}_t, \\mathrm{clip}(r_t, 1-\\epsilon, 1+\\epsilon) \\hat{A}_t)\\). For positive advantage, we cap the objective at \\((1+\\epsilon)\\hat{A}_t\\); for negative, we cap at \\((1-\\epsilon)\\hat{A}_t\\).
- When they differ: when \\(r_t > 1+\\epsilon\\) and \\(\\hat{A}_t > 0\\) (or \\(r_t < 1-\\epsilon\\) and \\(\\hat{A}_t < 0\\)). In those cases, the gradient from the clipped term is zero, so we do not update further in that direction.

**Common pitfalls**

- **Taking the min of two objectives:** The full PPO loss often combines \\(L^{CLIP}\\) with a value loss and an entropy bonus. The "min" in \\(L^{CLIP}\\) is between the ratio and the clipped ratio, not between two separate losses.
- **Clip range too small:** If \\(\epsilon = 0.05\\), updates are very small and learning can be slow. Typical \\(\epsilon\\) is 0.1–0.3.

{{< collapse summary="Worked solution (warm-up: PPO clip)" >}}
**Key idea:** PPO clips the probability ratio \\(r_t = \\pi(a_t|s_t)/\\pi_{old}(a_t|s_t)\\) to \\([1-\\epsilon, 1+\\epsilon]\\) so that the policy doesn’t change too much in one update. The objective is \\(\\min(r_t \\hat{A}_t, \\text{clip}(r_t, 1-\\epsilon, 1+\\epsilon) \\hat{A}_t)\\); we take the minimum so we are pessimistic when the advantage is positive. This keeps updates stable without a separate KL constraint like TRPO.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** If \\(r_t = 2\\) and \\(\\hat{A}_t = 0.5\\), and \\(\\epsilon = 0.2\\), what is the clipped objective value? What would the unclipped value be?
2. **Coding:** Write a function `ppo_clip_loss(ratio, advantage, eps=0.2)` that returns the per-sample clipped objective (the term inside the expectation). Test with ratio=1.5, advantage=1.0 and with ratio=0.5, advantage=-0.5.
3. **Challenge:** Plot the clipped objective as a function of \\(r_t\\) for fixed \\(\\hat{A}_t = 1\\) and \\(\\epsilon = 0.2\\). At what \\(r_t\\) does the gradient (with respect to \\(r_t\\)) become zero?
