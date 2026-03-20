---
title: "Chapter 76: Inverse Reinforcement Learning (IRL)"
description: "Max-ent IRL: learn reward from expert; linear reward, forward RL."
date: 2026-03-10T00:00:00Z
weight: 76
draft: false
difficulty: 8
tags: ["inverse RL", "IRL", "reward learning", "max entropy", "curriculum"]
keywords: ["inverse reinforcement learning", "IRL", "max entropy", "reward from expert"]
roadmap_color: "violet"
roadmap_icon: "database"
roadmap_phase_label: "Vol 8 · Ch 6"
---

**Learning objectives**

- **Implement** maximum entropy IRL: given expert trajectories, **learn a reward function** such that the expert's policy (approximately) maximizes expected return under that reward.
- **Use** a **linear reward** model (e.g. r(s, a) = w^T φ(s, a)) and **forward RL** (e.g. value iteration or policy gradient) to compute the optimal policy for the current reward.
- **Iterate** between updating the reward to make the expert look better than other policies and solving the forward RL problem.
- **Explain** why IRL can recover a reward that explains the expert behavior and then generalize (e.g. to new states) better than pure BC in some settings.
- **Relate** IRL to **robot navigation** (recovering intent from demonstrations) and **healthcare** (inferring treatment objectives).

**Concept and real-world RL**

**Inverse reinforcement learning (IRL)** infers a **reward function** from expert demonstrations, under the assumption that the expert is (approximately) optimal. **Maximum entropy IRL** posits that the expert acts to maximize return under some reward while being uncertain (max entropy) over paths with the same return. Given a reward parameterization (e.g. linear in features), we alternate: (1) solve **forward RL** to get the optimal policy for the current reward; (2) update the reward so that the expert's feature expectations match (or exceed) the optimal policy's. In **robot navigation** and **healthcare**, we may want to recover the underlying objective (e.g. "avoid obstacles," "minimize side effects") so we can generalize to new situations.

**Where you see this in practice:** Max-ent IRL, apprenticeship learning; reward learning from demonstrations.

**Illustration (learned reward):** MaxEnt IRL recovers a reward function that makes the expert optimal. The chart below shows expert return vs agent trained with learned reward (forward RL) over IRL iterations.

{{< chart type="line" palette="return" title="Agent return (trained on learned reward)" labels="IRL iter 1, Iter 2, Iter 3, Iter 4" data="50, 120, 200, 240" xLabel="IRL iteration" yLabel="Return" >}}

**Exercise:** Implement maximum entropy IRL: given expert trajectories, learn a reward function such that the expert's policy maximizes it. Use a linear reward model and solve the forward RL problem repeatedly.

**Professor's hints**

- **Linear reward:** r(s, a) = w^T φ(s, a). φ can be state-action features (e.g. coordinates, velocity). Compute expert feature counts: μ_expert = (1/N) sum over expert trajectories of sum_t φ(s_t, a_t). For a policy π, compute μ_π (same thing with rollouts under π).
- **Max-ent IRL (simplified):** Update w so that μ_expert has higher (discounted) reward than μ_π. One approach: gradient step on w to maximize (w^T μ_expert - w^T μ_π) or to maximize likelihood of expert under a softmax policy in the reward. Alternatively: w such that μ_expert = μ_π for the optimal π under w (feature matching).
- **Forward RL:** Given w, you have a reward; run value iteration, policy iteration, or policy gradient to get π*. Then compute μ_π* and compare to μ_expert; update w and repeat.
- Use a **small** MDP (e.g. gridworld with few states) so that forward RL is exact and feature expectations are easy to compute.

**Common pitfalls**

- **Reward ambiguity:** Many reward functions can explain the same expert (e.g. all rewards that are constant along the expert path). Max-ent and regularization (e.g. prefer small ||w||) help. In practice, IRL often needs a prior or regularization on the reward.
- **Forward RL cost:** Each IRL iteration requires solving an MDP; for large state spaces this is expensive. Use tabular or small neural policies for the exercise.
- **Convergence:** IRL is often formulated as a two-player game; convergence can be tricky. Run for a fixed number of iterations and check that expert return under learned w is high.

{{< collapse summary="Worked solution (warm-up: IRL)" >}}
**Key idea:** IRL assumes the expert maximizes some unknown reward \\(r(s,a; w)\\). We learn \\(w\\) so that the expert gets higher return than other policies (e.g. max-margin: expert return minus max over policies). Alternately we maximize the likelihood of expert trajectories under a softmax over returns. Once we have \\(w\\), we can run RL with \\(r(s,a; w)\\) to recover a policy that mimics the expert.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why might learning a reward function from the expert (and then doing forward RL) generalize better than behavioral cloning when the test environment differs slightly from the training distribution?
2. **Coding:** On a 5×5 gridworld with goal at (4,4), get expert trajectories (e.g. shortest path). Use binary features per cell (1 if in cell). Implement linear reward IRL: w has one entry per cell; forward RL = value iteration. Iterate: update w so expert feature count matches optimal policy's (or use gradient ascent on w^T (μ_expert - μ_π)). Report learned w (which cells have high reward?) and policy.
3. **Challenge:** Implement **maximum entropy** IRL: the expert policy is proportional to exp(return under r). Use a small tabular MDP and compute the partition function (sum over paths) for the max-ent policy. Update w to maximize expert trajectory likelihood.
4. **Variant:** Add noise to 20% of expert trajectories (random actions replacing the expert's). How does the learned reward function degrade? Does max-ent IRL handle noisy demonstrations better than standard IRL?
5. **Debug:** IRL training converges but the recovered reward assigns high reward to every cell equally. The feature matching objective is minimized by setting all weights w to zero. Add an L2 regularizer on w (or a max-margin constraint) and explain why unconstrained IRL often has degenerate solutions.
6. **Conceptual:** IRL assumes the expert is *optimal* under the unknown reward. What happens when this assumption is violated — e.g. the expert is a human making occasional mistakes? Describe how reward ambiguity (multiple rewards that explain the same behavior) complicates the IRL problem.
