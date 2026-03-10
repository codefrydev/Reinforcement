---
title: "Chapter 46: Maximum Entropy RL"
description: "Max-entropy objective; why entropy encourages exploration."
date: 2026-03-10T00:00:00Z
weight: 46
draft: false
tags: ["max entropy", "SAC", "exploration", "entropy", "curriculum"]
keywords: ["maximum entropy RL", "entropy bonus", "exploration", "SAC"]
---

**Learning objectives**

- Derive or state the **maximum entropy** objective: maximize \\(\\mathbb{E}[ \\sum_t r_t + \\alpha \\mathcal{H}(\\pi(\\cdot|s_t)) ]\\) (or equivalent), where \\(\\mathcal{H}\\) is entropy.
- Explain how the **entropy term** encourages exploration: higher entropy means more uniform action distribution, so the policy tries more actions.
- Contrast with **standard** expected return maximization (no entropy bonus).

**Concept and real-world RL**

**Maximum entropy RL** adds an entropy bonus to the objective so the agent maximizes return *and* policy entropy. The optimal policy under this objective is more stochastic (explores more) and is often easier to learn (multiple modes, robustness). In **robot control**, **SAC** (Soft Actor-Critic) uses this idea with automatic temperature tuning; in **game AI** and **recommendation**, entropy regularization (e.g. in PPO) prevents the policy from becoming too deterministic too fast. The temperature \\(\\alpha\\) (or equivalent) controls the trade-off between return and entropy.

**Where you see this in practice:** SAC, soft Q-learning, and PPO’s entropy bonus all relate to maximum entropy RL.

**Exercise:** Derive the maximum entropy objective and explain how it differs from standard expected return maximization. Why does entropy encourage exploration?

**Professor's hints**

- Standard: max \\(J = \\mathbb{E}[ \\sum_t \\gamma^t r_t ]\\). Max-ent: max \\(J + \\alpha \\mathbb{E}[ \\sum_t \\mathcal{H}(\\pi(\\cdot|s_t)) ]\\), where \\(\\mathcal{H}(\\pi) = -\\sum_a \\pi(a) \\log \\pi(a)\\) (discrete) or the continuous analogue.
- Exploration: \\(\\mathcal{H}\\) is maximized when \\(\\pi\\) is uniform; so the bonus pushes the policy toward trying all actions. In continuous space, a Gaussian with large std has high entropy.
### Derivation

Start from the Bellman equation with entropy; the optimal policy has a closed form in the linear case (softmax of Q/α). See Haarnoja et al., "Soft Actor-Critic."

**Common pitfalls**

- **Confusing entropy of policy with entropy of state distribution:** We add \\(\\mathcal{H}(\\pi(\\cdot|s))\\) (entropy of the action distribution given state), not the entropy of the state visitation distribution.
- **Alpha too large:** If \\(\\alpha\\) is huge, the agent ignores reward and just maximizes entropy (random policy). Tune \\(\\alpha\\) or use automatic tuning (SAC).

{{< collapse summary="Worked solution (warm-up: entropy coefficient)" >}}
**Key idea:** In maximum entropy RL we maximize return plus \\(\\alpha \\mathcal{H}(\\pi)\\). \\(\\alpha\\) controls the trade-off: large \\(\\alpha\\) favors random policies (high entropy); small \\(\\alpha\\) favors greedy. SAC uses a target entropy (e.g. \\(-\\dim(\\text{action})\\)) and tunes \\(\\alpha\\) automatically so the policy’s entropy tracks the target. That keeps exploration without hand-tuning \\(\\alpha\\).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For a discrete policy with two actions and \\(\\pi(a_1)=p\\), write the entropy \\(\\mathcal{H}\\) as a function of \\(p\\). At what \\(p\\) is \\(\\mathcal{H}\\) maximum?
2. **Coding:** In a 2-armed bandit, implement a "soft" policy that maximizes \\(\\mathbb{E}[r] + \\alpha \\mathcal{H}(\\pi)\\). Vary \\(\\alpha\\) and plot the probability of the optimal arm vs \\(\\alpha\\) (should approach 1 as \\(\\alpha \\to 0\\) and 0.5 as \\(\\alpha \\to \\infty\\)).
3. **Challenge:** Derive the optimal policy for a one-step MDP with max-ent objective: \\(\\pi^*(a|s) \\propto \\exp(Q(s,a)/\\alpha)\\). Show that as \\(\\alpha \\to 0\\) you get the greedy policy.
