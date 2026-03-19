---
title: "Chapter 6: The Bellman Equations"
description: "Derive Bellman optimality equation for Q*(s,a)."
date: 2026-03-10T00:00:00Z
weight: 6
draft: false
tags: ["Bellman equation", "value function", "MDP", "curriculum"]
keywords: ["Bellman equation", "value function", "Bellman expectation", "MDP"]
---

**Learning objectives**

- Derive the Bellman optimality equation for \\(Q^*(s,a)\\) from the definition of optimal action value.
- Contrast the optimality equation (max over actions) with the expectation equation (average over actions under \\(\\pi\\)).
- Explain why the optimality equations are nonlinear and how algorithms (e.g. value iteration) handle them.

**Concept and real-world RL**

The **optimal action-value function** \\(Q^*(s,a)\\) is the expected return from state \\(s\\), taking action \\(a\\), then acting optimally. The **Bellman optimality equation** for \\(Q^*\\) states that \\(Q^*(s,a)\\) equals the expected immediate reward plus \\(\\gamma\\) times the *maximum* over next-state action values (not an average under a policy). This "max" makes the system nonlinear: the optimal policy is greedy with respect to \\(Q^*\\), and \\(Q^*\\) is the fixed point of this equation. Value iteration and Q-learning are built on this; in practice, we approximate \\(Q^*\\) with tables or function approximators.

**Illustration (optimal action values):** In the two-state MDP from Chapter 3, state A has two actions (stay, go). The optimal action-value function \\(Q^*(A,\\text{stay})\\) and \\(Q^*(A,\\text{go})\\) differ; the chart below shows example values—the optimal policy chooses the action with the higher \\(Q^*\\).

{{< chart type="bar" title="Q*(A,a) for state A" labels="stay, go" data="4.5, 3.2" >}}

**Exercise:** Derive the Bellman optimality equation for action values, \\(Q^*(s,a)\\). Explain why it forms a system of nonlinear equations.

**Professor's hints**

- Start from the definition: \\(Q^*(s,a) = \\mathbb{E}[R_{t+1} + \\gamma \\max_{a'} Q^*(S_{t+1}, a') \\mid S_t=s, A_t=a]\\). The expectation is over \\(s', r\\) given \\(s, a\\); the max is over actions in the next state.
- Write it as \\(Q^*(s,a) = \\sum_{s',r} P(s',r|s,a) [r + \\gamma \\max_{a'} Q^*(s', a')]\\). The nonlinearity comes from \\(\\max_{a'}\\)—it is not a linear combination of \\(Q^*\\) values.
- For the "why nonlinear" part: the Bellman *expectation* equation for \\(V^\\pi\\) is linear in \\(V^\\pi\\) because \\(\\pi\\) is fixed. The optimality equation has a max, so the equation for \\(Q^*(s,a)\\) involves \\(\\max\\) of other \\(Q^*\\) values, which is nonlinear.

**Common pitfalls**

- **Writing expectation of max instead of max of expectation:** The correct form is \\(\\mathbb{E}[r + \\gamma \\max_{a'} Q^*(s',a')]\\), not \\(\\max_{a'} \\mathbb{E}[r + \\gamma Q^*(s',a')]\\). The max is inside the expectation over the *next* state.
- **Confusing \\(V^*(s)\\) with \\(\\max_a Q^*(s,a)\\):** They are equal: \\(V^*(s) = \\max_a Q^*(s,a)\\). So you can also write the equation in terms of \\(V^*\\) and then recover \\(Q^*\\).
- **Claiming the system is linear:** It is not; fixed-point iteration (value iteration) or Q-iteration is used instead of one-shot matrix inversion.

{{< collapse summary="Worked solution (Bellman optimality for Q*)" >}}
**Derivation:** By definition, \\(Q^*(s,a)\\) is the expected return from \\(s\\), taking \\(a\\), then acting optimally. So the return is \\(R_{t+1} + \gamma G_{t+1}\\), where \\(G_{t+1}\\) is the return from \\(S_{t+1}\\) under the optimal policy, i.e. \\(V^*(S_{t+1}) = \max_{a'} Q^*(S_{t+1}, a')\\). Taking expectation over \\(s', r\\) given \\(s, a\\):

\\(Q^*(s,a) = \sum_{s',r} P(s',r|s,a)\bigl[r + \gamma \max_{a'} Q^*(s', a')\bigr]\\).

**Why nonlinear:** The Bellman *expectation* equation for \\(V^\pi\\) is linear in \\(V^\pi\\) because the policy \\(\pi\\) is fixed. Here we have \\(\max_{a'}\\) inside the right-hand side, so \\(Q^*(s,a)\\) depends on the *maximum* of other \\(Q^*\\) values, which is a nonlinear operation. So we cannot write this as a linear system \\(Q = A Q\\); we use fixed-point iteration (value iteration, Q-iteration) instead of matrix inversion.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Write the Bellman optimality equation for \\(V^*(s)\\) (state value). Express it using \\(\\max_a\\) and \\(Q^*(s,a)\\) or \\(P(s',r|s,a)\\).
2. **Coding:** Implement value iteration for the 2-state MDP: loop \\(V_{k+1}(s) = \\max_a \\sum_{s',r} P(s',r|s,a)[r + \\gamma V_k(s')]\\) until \\(\\|V_{k+1} - V_k\\| < \\epsilon\\). Return \\(V^*\\) and the greedy policy.
3. **Challenge:** For the 2-state MDP from Chapter 3, write the two equations for \\(Q^*(A,\\text{stay})\\) and \\(Q^*(A,\\text{go})\\) in terms of \\(Q^*(B,\\cdot)\\) and \\(V^*(B)\\). Do not solve—just write the system.
4. **Variant:** Change the transition probabilities to \\(P(A \\to A | \\text{stay}) = 0.5\\) instead of 0.8. How do \\(Q^*(A,\\text{stay})\\) and \\(Q^*(A,\\text{go})\\) change? Does the optimal policy change?
5. **Debug:** The value iteration below incorrectly handles the terminal state — it updates its value instead of keeping it fixed. Find and fix the error.

{{< pyrepl code="# 2-state MDP: state 0=A (non-terminal), state 1=B (terminal)\nP = {0: {'stay': [(0, 1, 0.8), (1, 1, 0.2)],\n           'go':   [(1, 0, 1.0)]},\n     1: {}}  # terminal\nV = [0.0, 0.0]\ngamma = 0.9\nfor _ in range(50):\n    new_V = [0.0, 0.0]\n    for s in [0, 1]:  # BUG: should skip terminal state 1\n        if not P[s]:\n            new_V[s] = 0.0\n            continue\n        new_V[s] = max(\n            sum(prob*(r + gamma*V[ns]) for (ns,r,prob) in outcomes)\n            for outcomes in P[s].values()\n        )\n    V = new_V\nprint('V*:', V)" height="280" >}}

6. **Conceptual:** Why is the Bellman optimality equation nonlinear while the Bellman expectation equation (for a fixed policy) is linear?
7. **Recall:** State the Bellman optimality equation for \\(Q^*(s,a)\\) from memory.
