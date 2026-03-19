---
title: "Chapter 9: Dynamic Programming — Value Iteration"
description: "Value iteration on 4×4 gridworld, optimal V and policy."
date: 2026-03-10T00:00:00Z
weight: 9
draft: false
tags: ["value iteration", "dynamic programming", "optimal policy", "gridworld", "curriculum"]
keywords: ["value iteration", "optimal V", "optimal policy", "dynamic programming"]
---

**Learning objectives**

- Implement value iteration: repeatedly apply the Bellman optimality update for \\(V\\).
- Extract the optimal policy as greedy with respect to the converged \\(V\\).
- Relate value iteration to policy iteration (one sweep of "improvement" per state, no full evaluation).

**Concept and real-world RL**

**Value iteration** updates the state-value function using the Bellman *optimality* equation: \\(V(s) \\leftarrow \\max_a \\sum_{s',r} P(s',r|s,a)[r + \\gamma V(s')]\\). It does not maintain an explicit policy; after convergence, the optimal policy is greedy with respect to \\(V\\). Value iteration is simpler than full policy iteration (no inner evaluation loop) and converges to \\(V^*\\). It is used in planning when the model is known; in large or continuous spaces we approximate \\(V\\) or \\(Q\\) with function approximators and use approximate dynamic programming or model-free methods.

**Illustration (value iteration convergence):** The optimal value at a non-terminal state (e.g. center of the grid) updates each sweep and converges to \\(V^*\\). The chart below shows the value at one such state over the first few sweeps (γ=0.9, -1 per step).

{{< chart type="line" palette="return" title="V(s) at center cell over value iteration sweeps" labels="1, 2, 3, 4, 5, 6, 7, 8" data="-4, -3.5, -3.1, -2.8, -2.6, -2.4, -2.3, -2.2" xLabel="Sweep" yLabel="V(s)" >}}

**Exercise:** Implement value iteration for the same gridworld. Use \\(\gamma=0.9\\) and stop when the value function changes by less than 1e-4. Output the optimal value and policy.

**Professor's hints**

- Initialize \\(V(s)=0\\) for all states (including terminals). In each sweep, for every *non-terminal* state compute \\(V_{new}(s) = \\max_a \\sum_{s',r} P(s',r|s,a)[r + \\gamma V(s')]\\). Use the *current* \\(V\\) for all next-state values (synchronous update).
- Terminal states: keep \\(V\\) at 0 and do not update them. After the loop, extract the policy: for each state \\(s\\), \\(\\pi(s) = \\arg\\max_a \\sum_{s',r} P(s',r|s,a)[r + \\gamma V(s')]\\).
- Stop when \\(\\max_s |V_{new}(s) - V(s)| < 10^{-4}\\). Print or plot the 4×4 value table and a 4×4 policy (e.g. arrows or action indices).

**Common pitfalls**

- **Updating terminals:** Do not update \\(V\\) for terminal states; they stay 0. Updating them can break convergence or give wrong values.
- **Using \\(\\gamma=1\\) in the exercise:** The exercise specifies \\(\\gamma=0.9\\). With \\(\\gamma=1\\) and -1 per step, values can be very negative; the algorithm still converges but numbers differ.
- **Policy from V:** The optimal policy is greedy w.r.t. \\(V\\), not w.r.t. the *previous* policy. Compute the one-step lookahead using the *final* \\(V\\) when extracting the policy.

{{< collapse summary="Worked solution (warm-up: effect of γ on V)" >}}
**Warm-up:** Run value iteration with \\(\gamma=0.5\\) and then \\(\gamma=0.99\\) on the same gridworld. How does the optimal \\(V\\) near the center change?

**Answer:** With **higher \\(\gamma\\)** (e.g. 0.99), future rewards matter more, so the optimal \\(V\\) near the center is *less negative* (closer to 0 or higher) because the agent is “less discouraged” by the -1 per step—the distant terminal reward is weighted more. With **lower \\(\gamma\\)** (e.g. 0.5), the agent heavily discounts the future, so values near the center are more negative; the optimal policy may still go to the terminal, but the *magnitude* of \\(V\\) is smaller (more negative). So: higher \\(\gamma\\) ⇒ values reflect longer horizons and are typically larger (less negative when reward is -1 per step). This is the same trade-off in any discounted MDP: \\(\gamma\\) controls how much we care about the long term.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Run value iteration with \\(\\gamma=0.5\\) and then \\(\\gamma=0.99\\) on the same gridworld. How does the optimal \\(V\\) near the center change? (Higher \\(\\gamma\\) ⇒ future rewards matter more ⇒ values reflect longer horizons.)
2. **Coding:** Implement value iteration for a 4×4 gridworld with goal and terminal states. Plot the optimal \\(V^*\\) as a heatmap and the optimal policy as arrows. Verify that the policy points toward the goal.
3. **Challenge:** Add a single "obstacle" cell in the middle that the agent cannot enter (e.g. (1,1)). Update the transition model so actions into the obstacle leave the agent in place with reward -1. Re-run value iteration and show the optimal policy avoids the obstacle.
4. **Variant:** Run value iteration on a 6×6 gridworld with two goal cells at opposite corners, with rewards +1 and +5 respectively. How does the optimal policy change compared to a single-goal grid?
5. **Debug:** The code below updates \\(V\\) for terminal states (reward = 0, goal), causing the terminal's value to drift from 0. Find and fix it.

{{< pyrepl code="# 1D gridworld: states 0-4, terminal at 4 (goal)\nV = [0.0] * 5\ngamma = 0.9\nfor _ in range(30):\n    new_V = V[:]\n    for s in range(5):  # BUG: should skip state 4 (terminal)\n        if s < 4:\n            new_V[s] = max(-1 + gamma*V[s-1] if s>0 else -float('inf'),\n                           -1 + gamma*V[s+1] if s<4 else 1)\n        else:\n            new_V[s] = -1 + gamma*V[s]  # BUG: terminal updated\n    V = new_V\nprint('V:', [round(v,2) for v in V])" height="240" >}}

6. **Conceptual:** How does value iteration relate to policy iteration? In what sense is each VI sweep a combined evaluation + improvement step?
7. **Recall:** Write the value iteration update rule \\(V(s) \\leftarrow \\ldots\\) from memory.
