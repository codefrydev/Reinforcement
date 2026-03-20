---
title: "Chapter 71: The Offline RL Problem"
description: "Random policy dataset on Hopper; naive SAC overestimation."
date: 2026-03-10T00:00:00Z
weight: 71
draft: false
difficulty: 8
tags: ["offline RL", "Hopper", "SAC", "overestimation", "curriculum"]
keywords: ["offline RL", "random policy dataset", "SAC overestimation", "Hopper"]
roadmap_color: "violet"
roadmap_icon: "database"
roadmap_phase_label: "Vol 8 · Ch 1"
---

**Learning objectives**

- **Collect** a dataset of transitions (state, action, reward, next_state, done) from a **random policy** (or fixed behavior policy) in the Hopper environment.
- **Train** a standard SAC agent **offline** (no environment interaction) on this dataset and observe the **overestimation** of Q-values for out-of-distribution (OOD) actions.
- **Explain** why naive off-policy methods fail in offline RL: the policy is trained to maximize Q, but Q is only trained on in-distribution actions; for OOD actions Q can be overestimated.
- **Identify** the **distributional shift** between the behavior policy (that collected the data) and the learned policy.
- **Relate** the offline RL problem to **recommendation** and **healthcare** where data comes from logs or historical trials.

**Concept and real-world RL**

In **offline RL**, the agent learns from a **fixed dataset** of transitions (e.g. from a random or historical policy) without any environment interaction. **Naive** application of off-policy algorithms (e.g. SAC, DDPG) fails because the policy is optimized to choose actions that maximize Q, but Q-values for **out-of-distribution (OOD)** actions—actions that appear rarely or never in the data—can be **overestimated** due to extrapolation error. This **distributional shift** (learning a policy that deviates from the data distribution) leads to poor performance. In **recommendation** and **healthcare**, data often comes from logs or past trials; we cannot interact online, so offline RL and its pitfalls are directly relevant.

**Where you see this in practice:** Offline RL from logged data; batch RL; overestimation in DQN/SAC when used offline.

**Illustration (offline overestimation):** When training SAC on a fixed dataset without env interaction, Q-values can be overestimated for out-of-distribution actions. The chart below shows mean Q vs training steps (naive offline).

{{< chart type="line" palette="return" title="Mean Q(s,a) (offline SAC, no correction)" labels="0, 10k, 20k, 30k, 40k" data="2, 8, 25, 60, 120" xLabel="Step" yLabel="Mean Q(s,a)" >}}

**Exercise:** Collect a dataset from a random policy in the Hopper environment. Try to train a standard SAC agent offline (without environment interaction). Observe the overestimation issue and distributional shift.

**Professor's hints**

- **Dataset:** Run a random policy (or uniform action sampling) in Hopper for 1M steps (or similar); store (s, a, r, s', done) in a replay buffer and save to disk. Use a fixed seed for reproducibility.
- **Offline SAC:** Load the dataset into a replay buffer; do **not** step the environment. Train SAC as usual (sample batches from the buffer, update critic and actor). The actor will propose actions that may be OOD relative to the dataset.
- **Observe overestimation:** Log Q(s, a) for (s, a) from the dataset vs Q(s, π(s)) for the current policy. As training progresses, Q(s, π(s)) may become much larger than returns actually achievable in the env; then evaluate the policy in the env and see that actual return is low.
- Use a **small** dataset (e.g. 100k transitions) to make overestimation and failure more pronounced.

**Common pitfalls**

- **Evaluating with environment steps:** For a true offline experiment, evaluation should be limited (e.g. a few eval episodes) or use offline metrics (e.g. estimated return from the dataset). Avoid "peeking" with online data during training.
- **Data quality:** If the random policy rarely reaches good states, the dataset has little signal for high return; the agent may still overestimate but the gap between Q and actual return is the key observation.
- **Confusing on-policy vs off-policy:** SAC is off-policy, but it is designed for **online** data; the issue is that in offline RL we do not have data for the actions the learned policy would take.

{{< collapse summary="Worked solution (warm-up: offline RL)" >}}
**Key idea:** In offline RL we have a fixed dataset (no env interaction). The problem: the policy we are learning would take actions that may not appear in the data, so Q-values for those actions are overestimated (extrapolation error). Methods like CQL, BCQ, or TD3+BC add a penalty for OOD actions or constrain the policy to stay close to the behavior policy so we do not rely on out-of-distribution Q-values.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why can Q(s, a) be overestimated for an action a that rarely or never appears in the dataset?
2. **Coding:** Collect 50k transitions from a random policy on Hopper. Train SAC offline for 100k gradient steps. Plot mean Q(s, π(s)) on a held-out batch from the dataset vs actual evaluation return (e.g. 5 episodes every 10k steps). Do you see Q increasing while return stays low?
3. **Challenge:** Implement a simple **conservative** penalty: add a term to the Q-loss that penalizes Q(s, a) when a is from the current policy (not from the dataset). Compare with naive SAC on the same offline dataset.
4. **Variant:** Repeat the offline training with a medium-quality dataset (collected by a policy at 50% of expert performance) vs a random dataset. How much does dataset quality affect the severity of Q-value overestimation?
5. **Debug:** An offline SAC agent's evaluation return diverges to negative infinity after 50k steps. Logging shows Q-values grow unboundedly. The replay buffer samples only from in-distribution actions but the policy update uses out-of-distribution actions. Explain why this causes Q-value bootstrapping to explode and how a policy constraint would stabilize it.
6. **Conceptual:** Distribution shift in offline RL means the trained policy visits states not in the dataset. Why does this cause compounding errors? Use a brief analogy to supervised learning to explain why offline RL is harder than standard classification.
