---
title: "Chapter 5: Value Functions"
description: "State-value function V^π for random policy on Chapter 3 MDP."
date: 2026-03-10T00:00:00Z
weight: 5
draft: false
difficulty: 6
tags: ["value function", "state-value", "policy evaluation", "curriculum"]
keywords: ["value function", "V pi", "state-value function", "random policy"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Ch 5"
---

**Learning objectives**

- Define the state-value function \\(V^\\pi(s)\\) as the expected return from state \\(s\\) under policy \\(\\pi\\).
- Write and solve the Bellman expectation equation for a small MDP.
- Use matrix form (linear system) when the MDP is finite.

**Concept and real-world RL**

The **state-value function** \\(V^\\pi(s)\\) is the expected (discounted) return starting from state \\(s\\) and following policy \\(\\pi\\). It answers: "How good is it to be in this state if I follow this policy?" In games, \\(V(s)\\) is like the expected outcome from a board position; in navigation, it is the expected cumulative reward from a location. The **Bellman expectation equation** expresses \\(V^\\pi\\) in terms of immediate reward and the value of the next state; for finite MDPs it becomes a linear system \\(V = r + \\gamma P V\\) that we can solve by matrix inversion or iteration.

**Illustration (state values):** For the Chapter 3 two-state MDP under a random policy, solving the Bellman equation yields \\(V^\\pi(A)\\) and \\(V^\\pi(B)\\). The chart below shows example values for the two states.

{{< chart type="bar" title="V^π(s) for the two-state MDP (random policy)" labels="State A, State B" data="9, 10" >}}

**Exercise:** For the MDP in Chapter 3, compute the state-value function \\(V^\pi\\) for a random policy (\\(\pi(a|s)=0.5\\)) by solving the Bellman equation system manually (or via matrix inversion). Assume \\(\gamma = 0.9\\).

**Professor's hints**

- First write the Bellman equation for each state: \\(V^\\pi(s) = \\sum_a \\pi(a|s) \\sum_{s',r} P(s',r|s,a)[r + \\gamma V^\\pi(s')]\\). For the two-state MDP, you get two equations in two unknowns \\(V(A)\\) and \\(V(B)\\).
- Under a random policy, each action has probability 0.5. From state A you have two actions (stay, go); from B, two actions (both lead to A with reward -1). Write the expected immediate reward and the expected next-state value for each state.
- You can solve the 2×2 linear system by hand or with `np.linalg.solve`. In matrix form: \\(V = r^\\pi + \\gamma P^\\pi V\\) so \\((I - \\gamma P^\\pi) V = r^\\pi\\).

**Common pitfalls**

- **Wrong sign for rewards:** Rewards are part of the Bellman equation as \\(r + \\gamma V(s')\\). If your MDP defines "cost" as positive, use \\(-r\\) in the equation so that higher value means better.
- **Using the wrong policy:** The exercise asks for the *random* policy (\\(\\pi(a|s)=0.5\\) for each action). Do not use the optimal policy.
- **Forgetting discount:** \\(V(s)\\) includes \\(\\gamma\\) in front of the next-state value. If you write \\(V = r + V'\\) you are effectively using \\(\\gamma=1\\).

{{< collapse summary="Worked solution (warm-up: one-state MDP)" >}}
**Warm-up:** For a one-state MDP with one action that gives reward 1 and stays in the same state, write the Bellman equation and solve for \\(V\\).

**Step 1 — Bellman equation:** From the only state, we get reward 1 and stay. So \\(V = r + \\gamma V = 1 + \\gamma V\\).

**Step 2 — Solve:** \\(V - \\gamma V = 1\\), so \\(V(1 - \\gamma) = 1\\), hence \\(V = \\frac{1}{1-\\gamma}\\).

**Check:** For \\(\\gamma = 0.9\\), \\(V = 10\\) (infinite-horizon return \\(1 + 0.9 + 0.81 + \\cdots\\)). The same matrix form \\((I - \\gamma P^\\pi)V = r^\\pi\\) is used for the Chapter 3 two-state MDP with random policy.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For a one-state MDP with one action that gives reward 1 and stays in the same state, write the Bellman equation and solve for \\(V\\). (Answer: \\(V = 1 + \\gamma V\\) ⇒ \\(V = 1/(1-\\gamma)\\).)
2. **Coding:** In Python, encode the 2-state MDP from Chapter 3 (P, R matrices). Write a function that computes \\(V^\\pi\\) for a given policy using the Bellman expectation equation in matrix form (solve the linear system \\(V = R^\\pi + \\gamma P^\\pi V\\)).
3. **Challenge:** Write the Bellman equation for the *action-value* function \\(Q^\\pi(s,a)\\) for the Chapter 3 MDP and random policy. Express \\(Q^\\pi(A,\\text{stay})\\) and \\(Q^\\pi(A,\\text{go})\\) in terms of \\(V^\\pi(A)\\) and \\(V^\\pi(B)\\) (you can use your computed \\(V\\)).
4. **Variant:** Re-compute \\(V^\\pi\\) for the 2-state MDP with \\(\gamma=0.5\\) and \\(\gamma=0.99\\). How much do \\(V^\\pi(A)\\) and \\(V^\\pi(B)\\) change? Why does higher \\(\gamma\\) increase the magnitude of values?
5. **Debug:** The linear system below has a sign error (a common mistake). Find and fix it: \\((I + \\gamma P^\\pi) V = R^\\pi\\) — why is this wrong?

{{< pyrepl code="import numpy as np\n# 2-state MDP, random policy (0.5 each), gamma=0.9\n# States: 0=A, 1=B\nP_pi = np.array([[0.5*0.8 + 0.5*0,   0.5*0.2 + 0.5*1],\n                  [1.0,               0.0]])\nR_pi = np.array([0.5*1 + 0.5*0,   # state A: avg reward\n                  0.5*(-1) + 0.5*(-1)])  # state B\ngamma = 0.9\nI = np.eye(2)\n# Correct: (I - gamma*P_pi) V = R_pi\n# BUG below uses + instead of -\nV_buggy = np.linalg.solve(I + gamma * P_pi, R_pi)\nV_correct = np.linalg.solve(I - gamma * P_pi, R_pi)\nprint('Buggy V:', V_buggy)\nprint('Correct V:', V_correct)" height="260" >}}

6. **Conceptual:** Why is \\(V^\\pi\\) useful even when we do not have access to the transition model \\(P\\) (i.e. in model-free RL)?
7. **Recall:** State the Bellman expectation equation for \\(V^\\pi(s)\\) from memory.
