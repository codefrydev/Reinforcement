---
title: "Volume 5 Review & Bridge to Volume 6"
description: "Review Volume 5 (PPO, TRPO, SAC) and preview Volume 6 (Model-Based RL — learning world models and planning)."
date: 2026-03-19T00:00:00Z
draft: false
weight: 100
tags: ["review", "bridge", "Volume 5", "Volume 6", "PPO", "SAC", "model-based"]
---

## Volume 5 Recap Quiz (5 questions)

{{< collapse summary="Q1. What does PPO's clipped surrogate objective do, and why is it useful?" >}}
PPO clips the probability ratio r_t(θ) = π(a|s;θ) / π(a|s;θ_old) to stay within [1−ε, 1+ε]. The objective is: L_CLIP = E[min(r_t A_t, clip(r_t, 1−ε, 1+ε) A_t)]. This prevents the policy from moving too far from the old policy in a single update — without needing to solve a constrained optimization problem (unlike TRPO). It's simpler, faster, and works well in practice.
{{< /collapse >}}

{{< collapse summary="Q2. What is Generalized Advantage Estimation (GAE) and what does λ control?" >}}
GAE(λ) = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}, where δ_t = r_t + γV(s_{t+1}) − V(s_t) is the TD error. λ interpolates between: λ=0 → pure 1-step TD advantage (low variance, high bias); λ=1 → Monte Carlo advantage (high variance, low bias). Values like λ=0.95 give a good bias-variance tradeoff in practice.
{{< /collapse >}}

{{< collapse summary="Q3. What is the maximum entropy objective in SAC?" >}}
SAC maximises: J(π) = E[Σ_t r_t + α · H(π(·|s_t))], where H is the entropy of the policy and α is a temperature parameter. The entropy bonus encourages exploration and prevents premature convergence to deterministic policies. SAC is **off-policy** (uses a replay buffer) making it more sample-efficient than PPO, while the entropy regularization gives it robustness.
{{< /collapse >}}

{{< collapse summary="Q4. How does TRPO differ from PPO in enforcing the trust region?" >}}
TRPO solves a constrained optimization: maximize E[r_t A_t] subject to KL(π_old || π_new) ≤ δ. This requires computing the Fisher information matrix and solving a conjugate gradient problem — expensive. PPO approximates the same idea with the simple clip trick, avoiding second-order optimization. PPO is roughly as good as TRPO in practice but far simpler to implement.
{{< /collapse >}}

{{< collapse summary="Q5. When would you choose SAC over PPO?" >}}
**SAC** is preferable when: (1) the environment is expensive to simulate (SAC is more sample-efficient via replay); (2) the action space is continuous; (3) you want robustness to hyperparameters (entropy auto-tuning). **PPO** is preferable when: (1) the environment is fast to simulate (parallelism compensates for sample inefficiency); (2) discrete actions; (3) you want simplicity and proven stability across diverse tasks.
{{< /collapse >}}

---

## What Changes in Volume 6

| | Volume 5 (Model-Free) | Volume 6 (Model-Based) |
|---|---|---|
| **Environment model** | Black-box — just sample (s,a,r,s') | Learn ŝ' = f(s,a) and r̂ = r(s,a) |
| **Data efficiency** | Moderate to low | High — plan using the model |
| **Planning** | None | MCTS, Dyna-Q, shooting methods |
| **Risk** | Only real experience used | Model error compounds (hallucinations) |
| **Best for** | Fast simulators, complex rewards | Real-world, expensive interactions |

**The big insight:** If you know (or can learn) how the world works, you can *plan ahead* rather than only react. MCTS uses a model to search the game tree (AlphaGo). Dyna-Q uses a model to generate synthetic transitions. But learned models are imperfect — compounding errors over long rollouts is the central challenge.

---

## Bridge Exercise: When Would You Use a Model?

{{< pyrepl code="# When is model-based RL worth it?\n# Key question: is collecting real experience expensive?\n\nscenarios = [\n    {\"name\": \"Atari game\",         \"sim_cost\": \"free\",      \"use_model\": False},\n    {\"name\": \"Robot arm (real)\",    \"sim_cost\": \"very high\", \"use_model\": True},\n    {\"name\": \"Chess / Go\",          \"sim_cost\": \"free\",      \"use_model\": True},  # perfect model known\n    {\"name\": \"Drug discovery\",      \"sim_cost\": \"extreme\",   \"use_model\": True},\n    {\"name\": \"Video game (fast sim)\",\"sim_cost\": \"cheap\",    \"use_model\": False},\n]\n\nprint(f'{'Scenario':<25} {'Sim cost':<12} {'Model-based?'}')\nprint('-' * 50)\nfor s in scenarios:\n    rec = 'Yes -- plan!' if s['use_model'] else 'No  -- model-free'\n    print(f\"{s['name']:<25} {s['sim_cost']:<12} {rec}\")\n\nprint()\nprint('Rule of thumb:')\nprint('  Real-world / expensive sim  -> Model-based (Dyna, MBPO, MuZero)')\nprint('  Fast sim / complex rewards  -> Model-free (PPO, SAC)')\nprint('  Perfect known model (games) -> Planning (MCTS / AlphaZero)')" height="320" >}}

**Next:** [Volume 6: Model-Based RL](../volume-06/)
