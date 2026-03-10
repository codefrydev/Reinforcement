---
title: "Chapter 59: Probabilistic Ensembles with Trajectory Sampling (PETS)"
description: "PETS: ensemble dynamics, MPC with random shooting."
date: 2026-03-10T00:00:00Z
weight: 59
draft: false
---

**Learning objectives**

- Implement **PETS**: an **ensemble** of **probabilistic** dynamics models (e.g. output mean and variance), and **trajectory sampling** (e.g. random shooting or CEM) to select actions via **model predictive control (MPC)**.
- Use the model to evaluate action sequences and pick the best (no policy network).
- Apply to a **continuous control** task and compare with a policy-based method.

**Concept and real-world RL**

**PETS** uses an ensemble of probabilistic models to capture uncertainty; then at each step it **samples** many action sequences, rolls them out in the model, and chooses the sequence with the best predicted return (MPC). No policy network is trained; action selection is planning at test time. In **robot control**, MPC with learned models is used when we can afford computation at deployment; in **trading**, short-horizon planning with a learned model can improve decisions.

**Where you see this in practice:** PETS (Chua et al.); robotics MPC with learned models.

**Exercise:** Implement PETS for a continuous control task. Use an ensemble of probabilistic neural networks, and use trajectory sampling (e.g., random shooting) with the model to select actions via MPC.

**Professor's hints**

- Probabilistic model: output (mean, var) for next state (and reward). Train with negative log-likelihood. Ensemble: train K models; for rollout, sample one model per trajectory or sample from the ensemble prediction.
- Random shooting: sample N action sequences (e.g. H=10 steps, each action random or from a prior). Roll out each in the model; compute predicted sum of rewards. Pick the sequence with highest sum; execute first action (or first few); replan.
- CEM: alternatively, iteratively refine a distribution over action sequences by keeping the elite fraction and resampling.

**Common pitfalls**

- **Curse of dimensionality:** Random shooting over long horizons and high action dims is inefficient. Use short horizon (5–15) or CEM/cross-entropy method to focus samples.
- **Model uncertainty:** The ensemble gives different predictions; use the mean or sample one model per trajectory for consistency.

**Extra practice**

1. **Warm-up:** What is the difference between PETS and MBPO in terms of how the model is used (planning vs training a policy)?
2. **Coding:** Implement PETS for Pendulum with horizon H=10 and 500 random action sequences per step. Plot return vs step. How does it compare to SAC with the same number of env steps?
3. **Challenge:** Replace random shooting with **CEM**: maintain a Gaussian over action sequences; sample, evaluate, keep top 10%; update Gaussian; repeat for 5 iterations. Does CEM improve over random shooting?
