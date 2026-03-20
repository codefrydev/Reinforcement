---
title: "Chapter 46: Maximum Entropy RL"
description: "Max-entropy objective; why entropy encourages exploration."
date: 2026-03-10T00:00:00Z
weight: 46
draft: false
difficulty: 7
tags: ["max entropy", "SAC", "exploration", "entropy", "curriculum"]
keywords: ["maximum entropy RL", "entropy bonus", "exploration", "SAC"]
roadmap_color: "purple"
roadmap_icon: "rocket"
roadmap_phase_label: "Vol 5 · Ch 6"
---

**Learning objectives**

- Derive or state the **maximum entropy** objective: maximize \\(\\mathbb{E}[ \\sum_t r_t + \\alpha \\mathcal{H}(\\pi(\\cdot|s_t)) ]\\) (or equivalent), where \\(\\mathcal{H}\\) is entropy.
- Explain how the **entropy term** encourages exploration: higher entropy means more uniform action distribution, so the policy tries more actions.
- Contrast with **standard** expected return maximization (no entropy bonus).

**Concept and real-world RL**

**Maximum entropy RL** adds an entropy bonus to the objective so the agent maximizes return *and* policy entropy. The optimal policy under this objective is more stochastic (explores more) and is often easier to learn (multiple modes, robustness). In **robot control**, **SAC** (Soft Actor-Critic) uses this idea with automatic temperature tuning; in **game AI** and **recommendation**, entropy regularization (e.g. in PPO) prevents the policy from becoming too deterministic too fast. The temperature \\(\\alpha\\) (or equivalent) controls the trade-off between return and entropy.

**Where you see this in practice:** SAC, soft Q-learning, and PPO’s entropy bonus all relate to maximum entropy RL.

**Illustration (entropy over training):** Maximum entropy RL encourages exploration; policy entropy often starts high and may decrease as the policy becomes more deterministic. The chart below shows typical entropy over steps.

{{< chart type="line" palette="return" title="Policy entropy over training steps" labels="0, 20k, 40k, 60k, 80k" data="1.4, 1.0, 0.6, 0.4, 0.3" xLabel="Step" yLabel="Entropy" >}}

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
4. **Variant:** Try \\(\\alpha \\in \\{0.01, 0.1, 1.0, 5.0\\}\\) in your soft bandit. Plot the optimal-arm probability for each value. At what \\(\\alpha\\) does the policy become indistinguishable from uniform?

{{< pyrepl code="import math\n\ndef soft_policy(q_values, alpha):\n    \"\"\"Optimal max-entropy policy: proportional to exp(Q/alpha).\"\"\"\n    exp_q = [math.exp(q / alpha) for q in q_values]\n    total = sum(exp_q)\n    return [e / total for e in exp_q]\n\n# 2-armed bandit with Q(a1)=1.0, Q(a2)=0.5\nq = [1.0, 0.5]\nfor alpha in [0.01, 0.1, 1.0, 5.0]:\n    pi = soft_policy(q, alpha)\n    print(f'alpha={alpha:.2f}: pi(a1)={pi[0]:.3f}, pi(a2)={pi[1]:.3f}')" height="220" >}}

5. **Debug:** The code below confuses entropy of the policy with entropy of the state distribution, computing \\(-\\sum_s d(s) \\log d(s)\\) instead of \\(-\\sum_a \\pi(a|s) \\log \\pi(a|s)\\). Explain the difference.

```python
# BUG: computing entropy of state visitation, not action distribution
state_counts = {}
# ... accumulate state visit counts ...
entropy = -sum(p * math.log(p) for p in state_probs)  # wrong objective
# Fix: compute -sum_a pi(a|s) * log(pi(a|s)) for each state s
```

6. **Conceptual:** What happens to the optimal policy under the max-entropy objective as \\(\\alpha \\to 0\\)? As \\(\\alpha \\to \\infty\\)? Explain intuitively.
7. **Recall:** Write the maximum entropy RL objective \\(J_{max-ent} = \\mathbb{E}[\\ldots]\\) from memory and define \\(\\mathcal{H}(\\pi(\\cdot|s))\\).
