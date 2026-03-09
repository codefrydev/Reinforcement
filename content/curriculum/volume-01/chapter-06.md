---
title: "Chapter 6: The Bellman Equations"
description: "Derive Bellman optimality equation for Q*(s,a)."
date: 2026-03-10T00:00:00Z
weight: 6
draft: false
---

**Learning objectives**

- Derive the Bellman optimality equation for \\(Q^*(s,a)\\) from the definition of optimal action value.
- Contrast the optimality equation (max over actions) with the expectation equation (average over actions under \\(\\pi\\)).
- Explain why the optimality equations are nonlinear and how algorithms (e.g. value iteration) handle them.

**Concept and real-world RL**

The **optimal action-value function** \\(Q^*(s,a)\\) is the expected return from state \\(s\\), taking action \\(a\\), then acting optimally. The **Bellman optimality equation** for \\(Q^*\\) states that \\(Q^*(s,a)\\) equals the expected immediate reward plus \\(\\gamma\\) times the *maximum* over next-state action values (not an average under a policy). This "max" makes the system nonlinear: the optimal policy is greedy with respect to \\(Q^*\\), and \\(Q^*\\) is the fixed point of this equation. Value iteration and Q-learning are built on this; in practice, we approximate \\(Q^*\\) with tables or function approximators.

**Exercise:** Derive the Bellman optimality equation for action values, \\(Q^*(s,a)\\). Explain why it forms a system of nonlinear equations.

**Professor's hints**

- Start from the definition: \\(Q^*(s,a) = \\mathbb{E}[R_{t+1} + \\gamma \\max_{a'} Q^*(S_{t+1}, a') \\mid S_t=s, A_t=a]\\). The expectation is over \\(s', r\\) given \\(s, a\\); the max is over actions in the next state.
- Write it as \\(Q^*(s,a) = \\sum_{s',r} P(s',r|s,a) [r + \\gamma \\max_{a'} Q^*(s', a')]\\). The nonlinearity comes from \\(\\max_{a'}\\)—it is not a linear combination of \\(Q^*\\) values.
- For the "why nonlinear" part: the Bellman *expectation* equation for \\(V^\\pi\\) is linear in \\(V^\\pi\\) because \\(\\pi\\) is fixed. The optimality equation has a max, so the equation for \\(Q^*(s,a)\\) involves \\(\\max\\) of other \\(Q^*\\) values, which is nonlinear.

**Common pitfalls**

- **Writing expectation of max instead of max of expectation:** The correct form is \\(\\mathbb{E}[r + \\gamma \\max_{a'} Q^*(s',a')]\\), not \\(\\max_{a'} \\mathbb{E}[r + \\gamma Q^*(s',a')]\\). The max is inside the expectation over the *next* state.
- **Confusing \\(V^*(s)\\) with \\(\\max_a Q^*(s,a)\\):** They are equal: \\(V^*(s) = \\max_a Q^*(s,a)\\). So you can also write the equation in terms of \\(V^*\\) and then recover \\(Q^*\\).
- **Claiming the system is linear:** It is not; fixed-point iteration (value iteration) or Q-iteration is used instead of one-shot matrix inversion.

**Extra practice**

1. **Warm-up:** Write the Bellman optimality equation for \\(V^*(s)\\) (state value). Express it using \\(\\max_a\\) and \\(Q^*(s,a)\\) or \\(P(s',r|s,a)\\).
2. **Challenge:** For the 2-state MDP from Chapter 3, write the two equations for \\(Q^*(A,\\text{stay})\\) and \\(Q^*(A,\\text{go})\\) in terms of \\(Q^*(B,\\cdot)\\) and \\(V^*(B)\\). Do not solve—just write the system.
