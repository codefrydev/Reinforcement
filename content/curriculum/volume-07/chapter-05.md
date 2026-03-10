---
title: "Chapter 65: Count-Based Exploration"
description: "Count-based with hash table; pseudo-counts with density model for images."
date: 2026-03-10T00:00:00Z
weight: 65
draft: false
tags: ["count-based exploration", "pseudo-counts", "density model", "curriculum"]
keywords: ["count-based exploration", "hash table", "pseudo-counts", "density model"]
---

**Learning objectives**

- **Implement** count-based exploration for discrete state spaces using a hash table and a bonus such as \\(1/\\sqrt{N(s)}\\).
- **Implement** pseudo-counts from a density model (e.g. PixelCNN or simpler density estimator) for image-based states.
- **Explain** why pseudo-counts are needed when the state space is huge or continuous (e.g. Atari frames).
- **Test** count-based and pseudo-count exploration on a simple Atari-style or image-based task and compare exploration coverage.
- **Relate** count-based and pseudo-count methods to **game AI** and **recommendation** (e.g. diversity).

**Concept and real-world RL**

**Count-based exploration** gives a bonus for visiting states that have been seen fewer times; typically bonus \\(\\propto 1/\\sqrt{N(s)}\\) so that rarely visited states are more attractive. For **discrete** state spaces, \\(N(s)\\) can be stored in a hash table. For **high-dimensional or continuous** states (e.g. **game AI** with raw pixels), exact counts are impossible, so **pseudo-counts** are used: a density model is trained on visited states, and a pseudo-count is derived from the model's prediction (e.g. via the change in density after adding the state). In **recommendation**, similar diversity bonuses encourage exploring under-exposed items. This chapter ties tabular count-based ideas to deep RL with images.

**Where you see this in practice:** Count-based exploration in MDPs; pseudo-counts in Atari (e.g. Bellemare et al.); diversity bonuses in bandits and recommenders.

**Illustration (pseudo-count bonus):** Count-based exploration gives higher bonus to less-visited states. The chart below shows exploration bonus (e.g. \\(1/\\sqrt{N}\\)) for different visit counts.

{{< chart type="bar" title="Exploration bonus vs count" labels="N=1, N=5, N=10, N=20" data="1, 0.45, 0.32, 0.22" >}}

**Exercise:** For a discrete state space, implement count-based exploration using a hash table. Use pseudo-counts derived from a density model (e.g., a PixelCNN) for image-based states. Test on a simple Atari game.

**Professor's hints**

- **Discrete:** Maintain a dict or array for \\(N(s)\\); after each step, bonus = \\(1/\\sqrt{1+N(s)}\\), then \\(N(s) \\leftarrow N(s)+1\\). Use a hash (e.g. state tuple or flattened array) if the state is a grid or finite set.
- **Pseudo-counts for images:** A simple option is a **kernel density estimator** on a lower-dimensional embedding (e.g. from an autoencoder); or use a **PixelCNN** (or smaller density model) and derive pseudo-count from the model's predicted probability (see "Unifying count-based exploration" style papers). Start with a small input size (e.g. 42×42 grayscale) if using PixelCNN.
- For "simple Atari game," use a minimal env (e.g. one Atari game with frame stacking or a small custom image-based maze) so that training is feasible.
- Combine the exploration bonus with DQN or a policy-gradient method; add bonus to the reward before learning.

**Common pitfalls**

- **Hash collisions:** In discrete spaces with a hash, ensure the hash is consistent and that different states do not collide too often; use a proper hash of the state.
- **Pseudo-count implementation:** Pseudo-count formulas depend on the density model; use a standard derivation (e.g. based on learning progress or density ratio) to avoid ad hoc scaling.
- **Computational cost:** PixelCNN or large density models can be slow; start with a small model or lower-resolution frames.

{{< collapse summary="Worked solution (warm-up: density-based bonus)" >}}
**Key idea:** Fit a density model \\(p(s)\\) (e.g. Gaussian mixture, flow, or PixelCNN on images) on visited states. Intrinsic reward can be \\(-\\log p(s)\\) (surprise) or a pseudo-count derived from the density. Novel states have low \\(p(s)\\) and get high bonus. This generalizes to continuous and high-dimensional state spaces; the cost is training the density model.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For a state visited \\(N\\) times, the bonus is often \\(1/\\sqrt{N}\\). Why use the square root instead of \\(1/N\\)? (Think about the relative bonus for going from 1 to 2 visits vs 100 to 101.)
2. **Coding:** Implement hash-table count-based exploration on a 5×5 gridworld with 4 actions. Plot "unique states visited" and "episodes to reach goal" over 5000 episodes. Compare with ε-greedy (ε=0.1).
3. **Challenge:** Implement a simple **pseudo-count** using a small autoencoder: train it on visited states and use reconstruction error (or density in latent space) as a proxy for novelty. Use this as the exploration bonus in a simple image-based env and compare with RND.
