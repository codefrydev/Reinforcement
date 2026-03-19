---
title: "Chapter 73: Decision Transformers"
description: "Decision Transformer: returns-to-go, states, actions; GPT-like predict actions."
date: 2026-03-10T00:00:00Z
weight: 73
draft: false
tags: ["Decision Transformer", "returns-to-go", "offline RL", "curriculum"]
keywords: ["Decision Transformer", "returns-to-go", "GPT", "action prediction"]
---

**Learning objectives**

- **Implement** a Decision Transformer: a transformer (or GPT-style) model that takes sequences of (returns-to-go, state, action) and **predicts actions** conditioned on desired return (and past states/actions).
- **Explain** the formulation: at each timestep, input (R_t, s_t, a_{t-1}, R_{t-1}, s_{t-1}, ...) where R_t is the return from t onward; the model predicts a_t. Training is supervised on offline trajectories.
- **Train** the model on a simple environment's offline dataset and **test** by conditioning on different returns-to-go (e.g. high return for "expert" behavior).
- **Compare** with offline RL (e.g. CQL) in terms of implementation and how the policy is extracted (conditioning vs maximization).
- **Relate** Decision Transformers to **recommendation** (sequence of user-item-reward) and **dialogue** (conditioning on desired outcome).

**Concept and real-world RL**

**Decision Transformer (DT)** reframes RL as **sequence modeling**: instead of learning a value function or policy gradient, we treat a trajectory as a sequence of (returns-to-go, state, action) and train a transformer to **predict the action** given the desired return-to-go and history. At test time, we set the desired return-to-go (e.g. the maximum return seen in the dataset) and autoregressively generate actions. This avoids the overestimation problem of offline Q-learning because we do not bootstrap; we only do supervised learning on the dataset. In **recommendation** and **dialogue**, conditioning on "desired outcome" and generating actions (e.g. next item, next utterance) is a natural fit.

**Where you see this in practice:** Decision Transformer and variants; offline RL as sequence modeling; conditioning LLMs on return or reward.

**Illustration (return-to-go conditioning):** The Decision Transformer conditions on desired return-to-go; higher R leads to better behavior. The chart below shows actual return vs conditioned R_1 (e.g. 100, 200, 400).

{{< chart type="line" palette="return" title="Actual return vs conditioned R_1" labels="100, 200, 300, 400" data="80, 180, 280, 380" xLabel="Conditioned R₁" yLabel="Actual return" >}}

**Exercise:** Implement a Decision Transformer: treat trajectories as sequences of (returns-to-go, states, actions). Train a GPT-like model to predict actions conditioned on desired returns. Test on a simple environment.

**Professor's hints**

- **Returns-to-go:** For each timestep t, R_t = sum of rewards from t to end of trajectory. Compute these from the offline dataset for each trajectory.
- **Input format:** Option 1: (R_1, s_1, a_1, R_2, s_2, a_2, ...). Option 2: interleave as (R, s, a) tokens. The model sees past (R, s, a) and current R, s, and predicts a. Use embeddings for R, s, a (state can be a vector or CNN for pixels).
- **Training:** Cross-entropy (discrete actions) or MSE (continuous actions) for predicting a_t given (R_t, s_t) and history. Train on many trajectories; each trajectory gives multiple (prefix, target action) pairs.
- **Inference:** Start with desired R_1 (e.g. 90th percentile return in the dataset). Feed (R_1, s_1), get a_1; step env, get s_2, R_2 = R_1 - r_1; feed (R_2, s_2), get a_2; repeat.
- Use a **small** environment (e.g. CartPole or a small gridworld) and a small transformer (e.g. 2 layers, 4 heads) so training is fast.

**Common pitfalls**

- **Returns-to-go normalization:** If returns have very different scales across trajectories, normalize (e.g. z-score) or use a fixed scale so the model can generalize to "high" vs "low" return.
- **Context length:** The transformer has a limited context; for long episodes, truncate or use a sliding window. For short episodes (e.g. 100 steps), full context is fine.
- **Evaluation:** When testing, you must provide returns-to-go; if you always give the max return, the model should behave like the best trajectories in the dataset. Compare actual return when conditioning on max vs median return.

{{< collapse summary="Worked solution (warm-up: decision transformer)" >}}
**Key idea:** Decision Transformer conditions on (state, action, return-to-go) and predicts the next action. We train on offline trajectories; at test time we give a desired return-to-go (e.g. the max return in the dataset) and the model generates actions to achieve it. So we turn RL into supervised learning on sequences; the "return" is the conditioning signal that tells the model how well it should do.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why does predicting actions from (returns-to-go, state) avoid the overestimation problem of Q-learning in offline RL?
2. **Coding:** Implement a minimal DT on CartPole: collect 10k trajectories (mix of random and a trained policy). Train a small transformer to predict action from (R_t, s_t) and past. Evaluate by conditioning on R_1 = 400 (or max in data). Plot actual return vs R_1 used (e.g. 100, 200, 400).
3. **Challenge:** Use **continuous actions** (e.g. HalfCheetah): predict action with MSE loss. Handle variable-length trajectories by padding and masking. Compare with offline CQL on the same dataset.
4. **Variant:** Vary the context length K (number of past timesteps in the transformer's input) from 1 to 10 to 30. Does a longer context improve performance? On CartPole (Markovian), what do you expect vs a POMDP?
5. **Debug:** A Decision Transformer trained on CartPole evaluates to near-zero return when conditioned on R=400, but works fine for R=50. The model is trained with teacher forcing but never updates its return-to-go during evaluation (it keeps using the initial R throughout the episode instead of decrementing by each reward received). Fix the return-to-go update loop.
6. **Conceptual:** Decision Transformer frames RL as sequence modeling and does not use Bellman backups. What is the key advantage of this over CQL? What is the main disadvantage when the dataset does not contain high-return trajectories?
