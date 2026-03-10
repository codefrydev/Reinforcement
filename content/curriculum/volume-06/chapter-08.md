---
title: "Chapter 58: Model-Based Policy Optimization (MBPO)"
description: "MBPO: ensemble dynamics, short rollouts, SAC buffer."
date: 2026-03-10T00:00:00Z
weight: 58
draft: false
tags: ["MBPO", "ensemble dynamics", "SAC", "short rollouts", "curriculum"]
keywords: ["MBPO", "Model-Based Policy Optimization", "ensemble dynamics", "SAC"]
---

**Learning objectives**

- Implement **MBPO**: learn an **ensemble** of dynamics models, generate **short rollouts** from real states, add imagined transitions to the **replay buffer**, and train **SAC** on the combined buffer.
- Compare **sample efficiency** with **SAC alone** (same number of real env steps).
- Explain why short rollouts (e.g. 1–5 steps) help avoid compounding error.

**Concept and real-world RL**

**MBPO** (Model-Based Policy Optimization) uses learned dynamics to augment the replay buffer: from a real state, rollout the model for a few steps and add (s, a, r, s') to the buffer. SAC (or another off-policy method) then trains on real + imagined data. Short rollouts keep model error manageable. In **robot control** and **trading**, MBPO can significantly reduce the number of real steps needed to reach good performance.

**Where you see this in practice:** MBPO paper (Janner et al.); used in continuous control benchmarks.

**Illustration (MBPO vs SAC):** MBPO uses model-generated rollouts to augment the replay buffer; sample efficiency often improves. The chart below compares mean return vs env steps.

{{< chart type="line" palette="return" title="Mean return (MBPO vs SAC)" labels="0, 50k, 100k, 150k, 200k" data="0, 800, 2000, 3200, 4000" xLabel="Step" yLabel="Mean return" >}}

**Exercise:** Implement MBPO for a continuous task: learn an ensemble of dynamics models, use them to generate short rollouts from real states, and add these to the replay buffer for SAC training. Compare with SAC alone.

**Professor's hints**

- Ensemble: train 5–7 neural networks to predict (s', r) from (s, a). For rollout, pick a random model (or average predictions). Add Gaussian noise to predictions for diversity.
- Short rollouts: from each real s in the buffer (or a subset), sample an action from the current policy, predict s', r; add (s, a, r, s') to buffer. Do 1–5 steps. Do not rollout too long (compounding error).
- SAC: train as usual on the buffer (real + model-generated). Ratio of real to model data can be 1:4 or similar; tune.

**Common pitfalls**

- **Too long rollouts:** If you rollout 20 steps, the model's predictions drift; the data can be misleading. MBPO uses typically 1–5 steps.
- **Policy used for rollout:** Use the current policy to sample actions for model rollout so the data distribution is on-policy-ish for the policy being trained.

{{< collapse summary="Worked solution (warm-up: MBPO)" >}}
**Key idea:** MBPO trains a model on real data, then generates short imagined rollouts (e.g. 4–10 steps) from states in the buffer. We train the policy (e.g. SAC) on a mix of real and model-generated transitions. Short rollouts limit compounding error. This gives more data per env step and often reaches higher return faster than model-free alone. Compare by env steps (not by number of updates).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why does MBPO add only short model rollouts to the buffer?
2. **Coding:** Implement MBPO for Pendulum or Hopper. Plot return vs real env steps for MBPO and SAC. How much faster does MBPO reach a given return?
3. **Challenge:** Vary the rollout length (1, 3, 5, 10 steps). Plot final return vs rollout length. Is there an optimal length?
