---
title: "Chapter 56: MuZero Intuition"
description: "MuZero: model in latent space; reward prediction."
date: 2026-03-10T00:00:00Z
weight: 56
draft: false
---

**Learning objectives**

- Read a **MuZero** paper summary and explain how MuZero learns a **model in latent space** without access to the true environment dynamics.
- Explain how MuZero handles **reward prediction** and **value prediction** in the latent space.
- Contrast with AlphaZero (which uses the true game rules).

**Concept and real-world RL**

**MuZero** learns a latent dynamics model: instead of predicting raw next state, it predicts the next **latent state** and (optionally) **reward** and **value**. So the "model" is learned end-to-end for the purpose of planning; it does not need to match the true state. This allows MuZero to work in **video games** and **domains where rules are unknown**. In **game AI**, MuZero achieves strong results on Atari and board games without hand-coded dynamics.

**Where you see this in practice:** MuZero (DeepMind); applied to Atari, Go, chess.

**Exercise:** Read the MuZero paper summary. Explain how MuZero learns a model in latent space without access to the true dynamics. How does it handle the reward prediction?

**Professor's hints**

- Latent model: representation function h(s), dynamics function g(h,z,a) → next latent, reward head r(h), value head v(h). Train so that latent trajectory matches what would be useful for planning (e.g. TD target for value, observed reward for r).
- No true dynamics: the latent transition g is learned; we never predict the raw observation. So we can use it in MCTS without knowing the env.
- Reward: predict reward in latent space; train with observed reward. Value: predict value from latent; train with n-step return. Policy: from MCTS in the latent tree.

**Common pitfalls**

- **Confusing with world models:** MuZero's "model" is not a predictor of observations; it is a latent transition for planning. The loss is on value/reward/policy, not on state reconstruction.
- **Paper:** Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (2020).

**Extra practice**

1. **Warm-up:** What is the main difference between the dynamics in MuZero and the dynamics in a world model that predicts the next observation?
2. **Coding:** Sketch the MuZero architecture (representation, dynamics, reward, value, policy) in pseudocode or a diagram. Where does the MCTS run (in latent space)?
3. **Challenge:** In MuZero, why is it acceptable that the latent dynamics do not match the true environment? (Hint: what is the loss used to train them?)
