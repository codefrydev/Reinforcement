---
title: "Chapter 3: Markov Decision Processes (MDPs)"
description: "Two-state MDP transition probability matrices."
date: 2026-03-10T00:00:00Z
weight: 3
draft: false
tags: ["MDP", "Markov decision process", "transition probability", "curriculum"]
keywords: ["Markov decision process", "MDP", "transition probability", "state action reward"]
---

**Learning objectives**

- Define an MDP: states, actions, transition probabilities, and rewards.
- Write transition probability matrices \\(P(s' | s, a)\\) for a small MDP.
- Recognize the Markov property: the next state and reward depend only on the current state and action.

**Concept and real-world RL**

A **Markov Decision Process (MDP)** is the standard mathematical model for RL: a set of states, a set of actions, transition probabilities \\(P(s', r | s, a)\\), and a discount factor. The **Markov property** says that the future (next state and reward) depends only on the current state and action, not on earlier history. That allows us to plan using the current state alone. Real-world examples include board games (state = board position), robot navigation (state = position/velocity), and queue control (state = queue lengths). Writing out \\(P\\) and reward tables for a tiny MDP is the first step toward value iteration and policy iteration.

**Illustration (transition probabilities):** For the two-state MDP in the exercise, from state A with action "stay" the agent goes to A with probability 0.8 and to B with probability 0.2. The chart below shows this distribution over next states.

{{< chart type="bar" title="P(next state | A, stay)" labels="To A, To B" data="0.8, 0.2" >}}

**Exercise:** For a simple two-state MDP (states A, B) with actions: from A, action "stay" goes to A (prob 0.8) or B (0.2), reward +1; action "go" goes to B deterministically, reward 0. From B, both actions lead to A deterministically, reward -1. Write the transition probability matrices for each action.

**Professor's hints**

- You can use a matrix \\(P^a\\) where \\(P^a[i,j]\\) = probability of going from state \\(i\\) to state \\(j\\) under action \\(a\\). Order states consistently (e.g. row 0 = A, row 1 = B).
- For "stay" from A: P(A→A)=0.8, P(A→B)=0.2. For "go" from A: P(A→B)=1. Rewards are separate; the exercise asks for transition *probabilities* only, but keep rewards in mind for later (e.g. Bellman equations).
- From B, both actions give P(B→A)=1, P(B→B)=0. So the "from B" rows are the same for both actions.

**Common pitfalls**

- **Mixing up rows and columns:** Convention: row = current state, column = next state, so \\(P[i,j] = P(s'=j \\mid s=i)\\). Some texts use the transpose; pick one and stick to it.
- **Forgetting rewards:** The exercise asks for transition matrices, but in full MDPs you also have \\(r(s,a)\\) or \\(r(s,a,s')\\). When you move to value functions, you will need both.
- **Non-Markov state:** If you compress the state so that history matters (e.g. only "current cell" without "how many steps"), the process may not be Markov. For this two-state MDP, the state is explicit and Markov.

{{< collapse summary="Worked solution (transition matrices)" >}}
**Exercise:** Write the transition probability matrices for each action. Use states A = 0, B = 1; row = current state, column = next state.

**Step 1 — Action "stay":** From A: P(A→A)=0.8, P(A→B)=0.2. From B: P(B→A)=1, P(B→B)=0. So
\\(P^{\text{stay}} = \begin{bmatrix} 0.8 & 0.2 \\\\ 1 & 0 \end{bmatrix}\\).

**Step 2 — Action "go":** From A: P(A→B)=1, P(A→A)=0. From B: same as stay, P(B→A)=1, P(B→B)=0. So
\\(P^{\text{go}} = \begin{bmatrix} 0 & 1 \\\\ 1 & 0 \end{bmatrix}\\).

**Check:** Each row sums to 1. These matrices are the building blocks for the Bellman equation and value iteration in later chapters.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Write the 2×2 transition matrix for action "stay" (rows/cols = A, B). Verify each row sums to 1.
2. **Challenge:** Add a reward matrix \\(R(s,a)\\) (expected immediate reward). Write the Bellman expectation equation for \\(V^\\pi(A)\\) if \\(\\pi\\) always chooses "stay" in both states. Do not solve—just write the equation.
