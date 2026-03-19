---
title: "Chapter 32: The Policy Objective Function"
description: "Derive policy gradient theorem for one-step MDP."
date: 2026-03-10T00:00:00Z
weight: 32
draft: false
tags: ["policy gradient", "policy objective", "gradient theorem", "curriculum"]
keywords: ["policy objective function", "policy gradient theorem", "one-step MDP"]
---

**Learning objectives**

- Write the **policy gradient theorem** for a simple one-step MDP: the gradient of expected reward with respect to policy parameters.
- Show that \\(\nabla_\theta \mathbb{E}[R] = \mathbb{E}[ \nabla_\theta \log \pi(a|s;\theta) \, Q^\pi(s,a) ]\\) (or equivalent for one step).
- Recognize why this form is useful: we can estimate the expectation from samples (trajectories) without knowing the transition model.

**Concept and real-world RL**

In **policy gradient** methods we maximize the expected return \\(J(\theta) = \mathbb{E}_\pi[G]\\) by gradient ascent on \\(\theta\\). The **policy gradient theorem** says that \\(\nabla_\theta J\\) can be written as an expectation over states and actions under \\(\pi\\), involving \\(\nabla_\theta \log \pi(a|s;\theta)\\) and the return (or Q). For a **one-step MDP** (one state, one action, one reward), the derivation is simple: \\(J = \sum_a \pi(a|s) r(s,a)\\), so \\(\nabla_\theta J = \sum_a \nabla_\theta \pi(a|s) \, r(s,a)\\). Using the log-derivative trick \\(\nabla \pi = \pi \nabla \log \pi\\), we get \\(\mathbb{E}[ \nabla \log \pi(a|s) \, Q(s,a) ]\\). In **robot control** or **game AI**, we rarely have the full model; this identity lets us estimate the gradient from sampled actions and rewards only.

**Where you see this in practice:** The policy gradient theorem is the foundation for REINFORCE, actor-critic, and PPO. It appears in robotics (policy search), game playing, and dialogue systems.

**Illustration (gradient magnitude):** Policy gradient updates scale with the return; higher return trajectories get larger updates. The chart below shows the magnitude of \\(\\nabla \\log \\pi(a|s)\\) weighted by return over a few steps (conceptual).

{{< chart type="line" palette="return" title="|∇ log π · G| over trajectory steps" labels="Step 1, Step 2, Step 3, Step 4" data="0.5, 1.2, 2.1, 3.0" xLabel="Step" yLabel="|∇ log π · G|" >}}

**Exercise:** Derive the policy gradient theorem for a simple one-step MDP. Show that the gradient of the expected reward is \\(\mathbb{E}[\\nabla \\log \\pi(a|s) Q^\\pi(s,a)]\\).

**Professor's hints**

- One-step MDP: single state \\(s\\), agent samples \\(a \sim \pi(\cdot|s)\\), gets reward \\(r(s,a)\\). So \\(J(\theta) = \mathbb{E}_{a \sim \pi}[r(s,a)] = \sum_a \pi(a|s) r(s,a)\\).
- Log-derivative trick: \\(\nabla_\theta \pi(a|s) = \pi(a|s) \, \nabla_\theta \log \pi(a|s)\\). So \\(\nabla_\theta J = \sum_a \pi(a|s) \, \nabla_\theta \log \pi(a|s) \, r(s,a) = \mathbb{E}_\pi[ \nabla_\theta \log \pi(a|s) \, r(s,a) ]\\). For one step, \\(r(s,a) = Q^\pi(s,a)\\).
- In the multi-step case, \\(Q^\pi(s,a)\\) is replaced by the return from that step (e.g. \\(G_t\\)) in the full theorem.

**Common pitfalls**

- **Wrong sign:** We *maximize* \\(J\\), so the update is \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\), not minus. Loss-based frameworks often minimize \\(-J\\), in which case gradient descent on \\(-J\\) is equivalent.
- **Forgetting the expectation:** The theorem gives an expectation; in practice we use a *sample* (one trajectory or one action) to get an unbiased estimate of the gradient.

{{< collapse summary="Worked solution (warm-up: J(p) and policy gradient form)" >}}
**Warm-up:** \\(J(p) = p r_1 + (1-p) r_2\\). So \\(dJ/dp = r_1 - r_2\\). In policy gradient form: \\(\\nabla_\\theta J = \\mathbb{E}[ \\nabla_\\theta \\log \\pi(a|s) \\cdot r ]\\); for this one-step MDP, \\(\\nabla \\log \\pi(a_1|s) = 1/p\\) and \\(\\nabla \\log \\pi(a_2|s) = -1/(1-p)\\), so \\(\\mathbb{E}[ \\nabla \\log \\pi \\cdot r ] = p \\cdot (1/p) \\cdot r_1 + (1-p) \\cdot (-1/(1-p)) \\cdot r_2 = r_1 - r_2 = dJ/dp\\). This is the policy gradient theorem in the simplest case.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For a one-step MDP with two actions and \\(\pi(a_1|s)=p\\), \\(\pi(a_2|s)=1-p\\), and rewards \\(r_1, r_2\\), write \\(J(p)\\) and compute \\(dJ/dp\\) by hand. Then write it in the form \\(\mathbb{E}[ \nabla \log \pi \, r ]\\).
2. **Coding:** In Python, for a discrete policy \\(\pi = \mathrm{softmax}(\\theta)\\) with two actions, compute \\(\nabla_\theta \log \pi(a|s)\\) numerically (finite differences) and symbolically (derivative of log-softmax) and check they match.
3. **Challenge:** State the policy gradient theorem for the multi-step case (infinite horizon or episodic). What replaces \\(Q^\pi(s,a)\\) when we use Monte Carlo returns?
4. **Variant:** For the one-step MDP with rewards \\(r_1=1, r_2=-1\\), plot \\(J(p)\\) for \\(p \\in [0,1]\\) and mark where \\(dJ/dp=0\\). Does the optimal \\(p\\) match your gradient calculation?

{{< pyrepl code="import math\n\n# Log-derivative trick demo: one-step MDP, 2 actions\nr1, r2 = 1.0, -1.0\n\ndef J(p):\n    return p * r1 + (1 - p) * r2\n\ndef dJ_direct(p):\n    return r1 - r2  # analytical gradient\n\ndef dJ_log_trick(p, n_samples=10000):\n    \"\"\"Estimate dJ/dp via E[d/dp log pi(a) * r]\"\"\"\n    import random\n    total = 0\n    for _ in range(n_samples):\n        a = 0 if random.random() < p else 1\n        r = r1 if a == 0 else r2\n        # d/dp log pi(a): for a=0 it's 1/p, for a=1 it's -1/(1-p)\n        grad_log = 1/p if a == 0 else -1/(1-p)\n        total += grad_log * r\n    return total / n_samples\n\np = 0.4\nprint(f'Direct dJ/dp at p={p}:', dJ_direct(p))\nprint(f'Log-trick estimate:    ', round(dJ_log_trick(p), 2))" height="260" >}}

5. **Debug:** The gradient below has the wrong sign because it minimizes \\(J\\) instead of maximizing it. Fix the update direction.

```python
# BUG: gradient descent instead of gradient ascent on J
theta = theta - alpha * grad_log_pi * reward  # should be +
```

6. **Conceptual:** The log-derivative trick converts \\(\nabla_\theta \\pi(a|s;\\theta)\\) into \\(\\pi(a|s;\\theta) \\nabla_\theta \\log \\pi(a|s;\\theta)\\). Why is the log form more convenient for Monte Carlo estimation?
7. **Recall:** State the policy gradient theorem in the form \\(\\nabla_\\theta J(\\theta) = \\mathbb{E}[\\ldots]\\) from memory.
