---
title: "Chapter 32: The Policy Objective Function"
description: "Derive policy gradient theorem for one-step MDP."
date: 2026-03-10T00:00:00Z
weight: 32
draft: false
---

**Learning objectives**

- Write the **policy gradient theorem** for a simple one-step MDP: the gradient of expected reward with respect to policy parameters.
- Show that \\(\nabla_\theta \mathbb{E}[R] = \mathbb{E}[ \nabla_\theta \log \pi(a|s;\theta) \, Q^\pi(s,a) ]\\) (or equivalent for one step).
- Recognize why this form is useful: we can estimate the expectation from samples (trajectories) without knowing the transition model.

**Concept and real-world RL**

In **policy gradient** methods we maximize the expected return \\(J(\theta) = \mathbb{E}_\pi[G]\\) by gradient ascent on \\(\theta\\). The **policy gradient theorem** says that \\(\nabla_\theta J\\) can be written as an expectation over states and actions under \\(\pi\\), involving \\(\nabla_\theta \log \pi(a|s;\theta)\\) and the return (or Q). For a **one-step MDP** (one state, one action, one reward), the derivation is simple: \\(J = \sum_a \pi(a|s) r(s,a)\\), so \\(\nabla_\theta J = \sum_a \nabla_\theta \pi(a|s) \, r(s,a)\\). Using the log-derivative trick \\(\nabla \pi = \pi \nabla \log \pi\\), we get \\(\mathbb{E}[ \nabla \log \pi(a|s) \, Q(s,a) ]\\). In **robot control** or **game AI**, we rarely have the full model; this identity lets us estimate the gradient from sampled actions and rewards only.

**Where you see this in practice:** The policy gradient theorem is the foundation for REINFORCE, actor-critic, and PPO. It appears in robotics (policy search), game playing, and dialogue systems.

**Exercise:** Derive the policy gradient theorem for a simple one-step MDP. Show that the gradient of the expected reward is \\(\mathbb{E}[\\nabla \\log \\pi(a|s) Q^\\pi(s,a)]\\).

**Professor's hints**

- One-step MDP: single state \\(s\\), agent samples \\(a \sim \pi(\cdot|s)\\), gets reward \\(r(s,a)\\). So \\(J(\theta) = \mathbb{E}_{a \sim \pi}[r(s,a)] = \sum_a \pi(a|s) r(s,a)\\).
- Log-derivative trick: \\(\nabla_\theta \pi(a|s) = \pi(a|s) \, \nabla_\theta \log \pi(a|s)\\). So \\(\nabla_\theta J = \sum_a \pi(a|s) \, \nabla_\theta \log \pi(a|s) \, r(s,a) = \mathbb{E}_\pi[ \nabla_\theta \log \pi(a|s) \, r(s,a) ]\\). For one step, \\(r(s,a) = Q^\pi(s,a)\\).
- In the multi-step case, \\(Q^\pi(s,a)\\) is replaced by the return from that step (e.g. \\(G_t\\)) in the full theorem.

**Common pitfalls**

- **Wrong sign:** We *maximize* \\(J\\), so the update is \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\), not minus. Loss-based frameworks often minimize \\(-J\\), in which case gradient descent on \\(-J\\) is equivalent.
- **Forgetting the expectation:** The theorem gives an expectation; in practice we use a *sample* (one trajectory or one action) to get an unbiased estimate of the gradient.

**Extra practice**

1. **Warm-up:** For a one-step MDP with two actions and \\(\pi(a_1|s)=p\\), \\(\pi(a_2|s)=1-p\\), and rewards \\(r_1, r_2\\), write \\(J(p)\\) and compute \\(dJ/dp\\) by hand. Then write it in the form \\(\mathbb{E}[ \nabla \log \pi \, r ]\\).
2. **Coding:** In Python, for a discrete policy \\(\pi = \mathrm{softmax}(\\theta)\\) with two actions, compute \\(\nabla_\theta \log \pi(a|s)\\) numerically (finite differences) and symbolically (derivative of log-softmax) and check they match.
3. **Challenge:** State the policy gradient theorem for the multi-step case (infinite horizon or episodic). What replaces \\(Q^\pi(s,a)\\) when we use Monte Carlo returns?
