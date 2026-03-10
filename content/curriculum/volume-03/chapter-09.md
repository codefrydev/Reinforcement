---
title: "Chapter 29: Noisy Networks for Exploration"
description: "Noisy linear layers with factorized Gaussian; compare with ε-greedy."
date: 2026-03-10T00:00:00Z
weight: 29
draft: false
tags: ["NoisyNet", "exploration", "factorized Gaussian", "curriculum"]
keywords: ["Noisy Networks", "NoisyNet", "exploration", "epsilon-greedy"]
---

**Learning objectives**

- Implement **noisy linear layers**: \\(y = (W + \\sigma_W \\odot \\epsilon_W) x + (b + \\sigma_b \\odot \\epsilon_b)\\), where \\(\\epsilon\\) is random noise (e.g. Gaussian) and \\(\\sigma\\) are learnable parameters.
- Use **factorized Gaussian** noise to reduce the number of random samples: e.g. \\(\\epsilon_{i,j} = f(\\epsilon_i) \\cdot f(\\epsilon_j)\\) with \\(f\\) such that the product has zero mean and unit variance.
- Compare exploration (e.g. unique states visited, or variance of actions over time) with \\(\\epsilon\\)-greedy DQN.

**Concept and real-world RL**

**Noisy networks** add learnable noise to the weights (or activations) of the network. The noise is resampled every forward pass (or every step), so the policy is stochastic and explores without an explicit \\(\\epsilon\\)-greedy schedule. The scale of the noise can be learned (the network can reduce noise when it is confident). **Factorized Gaussian** generates a matrix of noise from two vectors (e.g. \\(\\epsilon_{out} \\otimes \\epsilon_{in}\\)) so we only need \\(O(n+m)\\) random numbers instead of \\(O(nm)\\). Noisy DQN is used in Rainbow and provides state-dependent exploration.

**Exercise:** Replace the linear layers in your DQN with noisy linear layers that have learnable noise. Implement the factorized Gaussian noise. Compare exploration behavior (e.g., number of unique states visited) with \\(\epsilon\\)-greedy.

**Professor's hints**

- Noisy Linear: store \\(W\\), \\(b\\) (mean) and \\(\\sigma_W\\), \\(\\sigma_b\\) (learnable scale). Each forward: sample \\(\\epsilon_W\\), \\(\\epsilon_b\\) (e.g. Gaussian or signed constant), then \\(y = (W + \\sigma_W \\odot \\epsilon_W) x + (b + \\sigma_b \\odot \\epsilon_b)\\). Use the same \\(\\epsilon\\) for the whole batch in one step (so exploration is consistent within the step).
- Factorized: for a layer with \\(in\\) and \\(out\\) features, sample \\(\\epsilon_{in}\\) (in,) and \\(\\epsilon_{out}\\) (out,). Then \\(\\epsilon_{i,j} = f(\\epsilon_{out,i}) \\cdot f(\\epsilon_{in,j})\\) where \\(f(x) = \\text{sign}(x) \\sqrt{|x|}\\) (for signed constant) or use \\(\\epsilon \\sim \\mathcal{N}(0,1)\\) and factorize. See the Noisy Networks paper for the exact formula.
- Comparison: run \\(\\epsilon\\)-greedy DQN and Noisy DQN (no \\(\\epsilon\\)) for the same number of steps. Log the number of unique (s,a) or unique states visited in the first N steps. Noisy often gives state-dependent exploration (more exploration in uncertain states).

**Common pitfalls**

- **Resampling every sample:** Noise should be resampled at least every environment step (or every forward for the policy), not once at init. Otherwise there is no exploration over time.
- **Gradient through noise:** Usually we do *not* backprop through the noise (noise is not differentiable). So \\(\\sigma\\) gets gradients from the loss, but \\(\\epsilon\\) is detached. The reparameterization trick (sample \\(\\epsilon\\), then \\(W + \\sigma \\odot \\epsilon\\)) gives gradients to \\(\\sigma\\).
- **Initialization of \\(\\sigma\\):** Start with small \\(\\sigma\\) (e.g. 0.5) so early training is not too noisy. The network can learn to reduce \\(\\sigma\\) later.

{{< collapse summary="Worked solution (warm-up: learnable σ in noisy nets)" >}}
**Warm-up:** In a noisy linear layer, why do we need *learnable* \\(\\sigma\\)? **Answer:** So the network can *reduce* exploration when it has learned. If \\(\\sigma\\) were fixed, we would always inject the same noise. With learnable \\(\\sigma\\), the network can drive \\(\\sigma \\to 0\\) in parts of the state space where it is confident, and keep larger \\(\\sigma\\) where it is uncertain. That gives state-dependent exploration without a separate \\(\\epsilon\\)-schedule.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In a noisy linear layer, why do we need *learnable* \\(\\sigma\\)? (So the network can reduce exploration when it has learned; state-dependent exploration.)
2. **Coding:** Implement a noisy linear layer: weight = mu + sigma * epsilon, epsilon ~ N(0,1), with learnable mu and sigma. Use it as the last layer of a DQN. Run 5k steps on CartPole with no ε-greedy (noise only) and plot return.
3. **Challenge:** Replace only the *last* layer with a noisy layer and keep the rest standard. Compare with full noisy DQN. Is most of the benefit from the last layer?
