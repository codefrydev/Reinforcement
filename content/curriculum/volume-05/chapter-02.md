---
title: "Chapter 42: Trust Region Policy Optimization (TRPO)"
description: "TRPO constrained optimization and natural gradient; KL constraint."
date: 2026-03-10T00:00:00Z
weight: 42
draft: false
tags: ["TRPO", "trust region", "KL constraint", "natural gradient", "curriculum"]
keywords: ["TRPO", "Trust Region Policy Optimization", "KL constraint", "natural gradient"]
---

**Learning objectives**

- Read and summarize the **TRPO** paper: the constrained optimization problem (maximize expected advantage subject to KL constraint between old and new policy).
- Explain why the **natural gradient** (using the Fisher information matrix) approximates the KL-constrained step.
- Relate the KL constraint to preventing too-large policy updates (connection to Chapter 41).

**Concept and real-world RL**

**TRPO** (Trust Region Policy Optimization) limits each policy update so that the new policy stays close to the old one in the sense of **KL divergence**: maximize \\(\mathbb{E}[ \\frac{\\pi(a|s)}{\\pi_{old}(a|s)} A^{old}(s,a) ]\\) subject to \\(\mathbb{E}[ D_{KL}(\\pi_{old} \\| \\pi) ] \\leq \\delta\\). This prevents the collapse and instability seen in vanilla policy gradients. The **natural gradient** (preconditioning by the Fisher information matrix) gives an approximate solution to this constrained problem. In **robot control** and **safety-critical** settings, TRPO’s monotonic improvement guarantee (under assumptions) is appealing; in practice **PPO** is often preferred for its simpler implementation (clipped objective instead of constrained optimization).

**Where you see this in practice:** TRPO is used in robotics and was a precursor to PPO. The trust-region idea appears in constrained RL and safe policy improvement.

**Illustration (KL constraint):** TRPO limits the KL divergence between old and new policy per update. The chart below shows typical KL(π_old || π_new) over TRPO iterations (stays below threshold).

{{< chart type="line" title="KL(π_old || π_new) per update" labels="Iter 1, Iter 2, Iter 3, Iter 4, Iter 5" data="0.02, 0.015, 0.012, 0.01, 0.008" >}}

**Exercise:** (Theoretical) Read the TRPO paper. Derive the constrained optimization problem and explain why the natural gradient step (using the Fisher information matrix) enforces the KL constraint.

**Professor's hints**

- Constrained problem: max \\(\mathbb{E}_{s,a \\sim \\pi_{old}}[ \\frac{\\pi(a|s)}{\\pi_{old}(a|s)} A(s,a) ]\\) s.t. \\(\bar{D}_{KL}(\\pi_{old} \\| \\pi) \\leq \\delta\\). The objective is the surrogate advantage; the constraint keeps \\(\pi\\) close to \\(\pi_{old}\\).
- Natural gradient: the Fisher information matrix \\(F\\) (expected outer product of \\(\nabla \\log \\pi\\)) defines a metric on the policy space. The natural gradient is \\(F^{-1} \\nabla J\\); taking a step in this direction (with appropriate step size) approximately respects the KL ball.
- Paper: Schulman et al., "Trust Region Policy Optimization" (2015). Focus on Section 2 (problem setup) and Section 3 (approximation and algorithm).

**Common pitfalls**

- **Conjugate gradient and line search:** Full TRPO uses conjugate gradient to compute \\(F^{-1} g\\) and a line search to satisfy the KL constraint. The "natural gradient" explanation is the intuition; the actual algorithm is more involved.
- **KL vs reverse KL:** TRPO constrains \\(D_{KL}(\\pi_{old} \\| \\pi)\\) (old relative to new). The direction matters for the optimization geometry.

{{< collapse summary="Worked solution (warm-up: TRPO constraint)" >}}
**Key idea:** TRPO maximizes the surrogate objective \\(\\mathbb{E}[ \\frac{\\pi(a|s)}{\\pi_{old}(a|s)} A ]\\) subject to \\(\\mathbb{E}_s[ D_{KL}(\\pi_{old}(\\cdot|s) \\| \\pi(\\cdot|s)) ] \\leq \\delta\\). The KL constraint keeps the new policy close to the old so the surrogate remains a good approximation. We solve the constrained problem with a conjugate gradient step plus line search. This avoids the collapse that large policy gradient steps can cause.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence, what is the role of the KL constraint in TRPO?
2. **Coding:** For a small policy (e.g. 2 actions, parameter \\(\theta\\)), compute the Fisher information matrix \\(F = \\mathbb{E}[ (\\nabla \\log \\pi)^T (\\nabla \\log \\pi) ]\\) numerically (sample actions from \\(\pi\\)). Verify it is positive definite.
3. **Challenge:** Implement a **simplified** TRPO step: one natural gradient step with a fixed step size that you tune so the mean KL after the update is roughly \\(\delta = 0.01\\). Compare with a vanilla policy gradient step of the same magnitude (in parameter space).
