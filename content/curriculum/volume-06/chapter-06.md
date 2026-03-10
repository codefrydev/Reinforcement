---
title: "Chapter 56: MuZero Intuition"
description: "MuZero: model in latent space; reward prediction."
date: 2026-03-10T00:00:00Z
weight: 56
draft: false
tags: ["MuZero", "latent space", "reward prediction", "model-based", "curriculum"]
keywords: ["MuZero", "latent model", "reward prediction", "model-based RL"]
---

**Learning objectives**

- Read a **MuZero** paper summary and explain how MuZero learns a **model in latent space** without access to the true environment dynamics.
- Explain how MuZero handles **reward prediction** and **value prediction** in the latent space.
- Contrast with AlphaZero (which uses the true game rules).

**Concept and real-world RL**

**MuZero** learns a latent dynamics model: instead of predicting raw next state, it predicts the next **latent state** and (optionally) **reward** and **value**. So the "model" is learned end-to-end for the purpose of planning; it does not need to match the true state. This allows MuZero to work in **video games** and **domains where rules are unknown**. In **game AI**, MuZero achieves strong results on Atari and board games without hand-coded dynamics.

**Where you see this in practice:** MuZero (DeepMind); applied to Atari, Go, chess.

**Illustration (latent model):** MuZero learns a model in latent space; the chart below shows conceptual reward prediction accuracy (train vs test) over training.

{{< chart type="line" palette="return" title="Reward prediction accuracy (MuZero-style)" labels="0, 10k, 20k, 30k, 40k" data="0.3, 0.6, 0.78, 0.88, 0.92" xLabel="Step" yLabel="Accuracy" >}}

**Exercise:** Read the MuZero paper summary. Explain how MuZero learns a model in latent space without access to the true dynamics. How does it handle the reward prediction?

**Professor's hints**

- Latent model: representation function h(s), dynamics function g(h,z,a) → next latent, reward head r(h), value head v(h). Train so that latent trajectory matches what would be useful for planning (e.g. TD target for value, observed reward for r).
- No true dynamics: the latent transition g is learned; we never predict the raw observation. So we can use it in MCTS without knowing the env.
- Reward: predict reward in latent space; train with observed reward. Value: predict value from latent; train with n-step return. Policy: from MCTS in the latent tree.

**Common pitfalls**

- **Confusing with world models:** MuZero's "model" is not a predictor of observations; it is a latent transition for planning. The loss is on value/reward/policy, not on state reconstruction.
- **Paper:** Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (2020).

{{< collapse summary="Worked solution (warm-up: MuZero)" >}}
**Key idea:** MuZero learns a latent model: \\(s_{t+1} = f(s_t, a_t)\\) and reward/value in latent space. We do not predict pixels; we plan in the latent space. The value and reward heads are trained with TD (or n-step) on imagined rollouts. So we get the benefit of planning with a model without needing to decode to observation space. This scales to Atari and board games.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** What is the main difference between the dynamics in MuZero and the dynamics in a world model that predicts the next observation?
2. **Coding:** Sketch the MuZero architecture (representation, dynamics, reward, value, policy) in pseudocode or a diagram. Where does the MCTS run (in latent space)?
3. **Challenge:** In MuZero, why is it acceptable that the latent dynamics do not match the true environment? (Hint: what is the loss used to train them?)
