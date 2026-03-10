---
title: "Chapter 74: Introduction to Imitation Learning"
description: "Expert demos from PPO on LunarLander; behavioral cloning."
date: 2026-03-10T00:00:00Z
weight: 74
draft: false
tags: ["imitation learning", "behavioral cloning", "LunarLander", "PPO", "curriculum"]
keywords: ["imitation learning", "behavioral cloning", "expert demos", "LunarLander"]
---

**Learning objectives**

- **Collect** expert demonstrations (state-action pairs or trajectories) from a **trained PPO agent** on LunarLander.
- **Train** a **behavioral cloning (BC)** agent: supervised learning to predict the expert's action given the state.
- **Evaluate** the BC policy in the environment and compare its return and behavior to the expert.
- **Explain** the assumptions of behavioral cloning (i.i.d. states from the expert distribution) and when it works well.
- **Relate** imitation learning to **robot navigation** (learning from human demos) and **dialogue** (learning from human responses).

**Concept and real-world RL**

**Imitation learning** aims to learn a policy from **expert demonstrations** (state-action pairs or trajectories) without using environment reward. **Behavioral cloning (BC)** is the simplest approach: treat it as supervised learning—given state s, predict the expert's action a. If the data distribution (states visited by the expert) matches the distribution the agent will see at test time, BC can work well. In **robot navigation**, experts can provide demonstrations via teleoperation; in **dialogue**, expert responses can be used to train a response policy. BC is easy to implement but can suffer from **covariate shift** when the agent visits states not well covered by the expert (see next chapter).

**Where you see this in practice:** Behavioral cloning for robotics and self-driving; learning from demonstrations (LfD); DAgger and inverse RL build on this.

**Illustration (BC vs expert):** Behavioral cloning mimics the expert from demonstrations. The chart below shows expert return vs BC agent return (same env) for different dataset sizes.

{{< chart type="bar" palette="comparison" title="Mean return (expert vs BC)" labels="Expert, BC (100 demos), BC (1000 demos)" data="250, 120, 220" yLabel="Mean return" >}}

**Exercise:** Collect expert demonstrations from a trained PPO agent on LunarLander. Train a behavioral cloning agent on this data. Evaluate how well it mimics the expert.

**Professor's hints**

- **Expert:** Train PPO (or any policy) on LunarLander until it achieves a good return (e.g. 200+). Save trajectories (s, a, r, s', done) from multiple episodes.
- **BC:** Train a policy network (same architecture as PPO actor, or simpler) with cross-entropy loss (discrete actions) or MSE (continuous) to predict a given s. Use the same state representation as the expert (e.g. raw state vector).
- **Evaluation:** Run the BC policy for 20–50 episodes; report mean and std of return. Compare with expert return. Also compare **state distribution**: do BC and expert visit similar states, or does BC drift?
- Collect at least 50–100 expert episodes so the BC agent has enough diversity.

**Common pitfalls**

- **Too few demos:** With very few trajectories, BC overfits and does not generalize; collect more episodes or add regularization.
- **Expert not good enough:** If the expert is suboptimal, BC will copy its mistakes; ensure the expert is reasonably strong.
- **Ignoring covariate shift:** BC is trained on expert state distribution; when the BC agent runs, it may make small errors, visit new states, and then get into regions where it has no training data and fail. We will address this with DAgger in the next chapter.

{{< collapse summary="Worked solution (warm-up: behavioral cloning)" >}}
**Key idea:** Behavioral cloning (BC) is supervised learning: given expert (s,a) pairs, train a policy \\(\\pi_\\theta(a|s)\\) to mimic the expert (e.g. cross-entropy or MSE). Simple and stable, but the agent only sees expert states; at test time it may drift to novel states and then fail (covariate shift). BC is a strong baseline for imitation; we improve with DAgger (query expert on agent states) or offline RL.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why is behavioral cloning a supervised learning problem? What is the "input" and "output"?
2. **Coding:** Collect 100 expert episodes from PPO on LunarLander. Train BC with 80% of the data; hold out 20% for validation (action prediction accuracy). Report validation accuracy and evaluation return. How many episodes are needed for BC to get within 80% of expert return?
3. **Challenge:** Add **data augmentation** to BC (e.g. add small noise to states, or use dropout). Does it improve evaluation return when you have limited demos (e.g. 20 episodes)?
