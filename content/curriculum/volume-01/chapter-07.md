---
title: "Chapter 7: Dynamic Programming — Policy Evaluation"
description: "Iterative policy evaluation on 4×4 gridworld."
date: 2026-03-10T00:00:00Z
weight: 7
draft: false
tags: ["dynamic programming", "policy evaluation", "gridworld", "curriculum"]
keywords: ["policy evaluation", "iterative policy evaluation", "dynamic programming", "gridworld"]
---

**Learning objectives**

- Implement iterative policy evaluation (Bellman expectation updates) for a finite MDP.
- Use a gridworld with terminal states and interpret the resulting value function.
- Decide when to stop iterating (e.g. max change below a threshold).

**Concept and real-world RL**

**Policy evaluation** computes \\(V^\\pi\\) for a given policy \\(\\pi\\). **Iterative policy evaluation** starts from an arbitrary \\(V\\) (e.g. zeros) and repeatedly applies the Bellman expectation update: \\(V(s) \\leftarrow \\sum_a \\pi(a|s) \\sum_{s',r} P(s',r|s,a)[r + \\gamma V(s')]\\). This converges to \\(V^\\pi\\) for finite MDPs. In a gridworld, values spread from terminal states (goal or trap); the result shows "how good" each cell is under the policy. This is the building block for policy iteration (evaluate, then improve the policy).

**Illustration (convergence):** As we apply the Bellman expectation update sweep after sweep, the maximum change in \\(V\\) typically decreases. The chart below shows a typical trend: max \\(|\\Delta V|\\) over the first few sweeps until convergence.

{{< chart type="line" palette="return" title="Max |ΔV| per sweep (policy evaluation)" labels="Sweep 1, Sweep 2, Sweep 3, Sweep 4, Sweep 5, Sweep 6" data="2.5, 1.2, 0.5, 0.15, 0.04, 0.01" xLabel="Sweep" yLabel="Max |ΔV|" >}}

**Exercise:** Implement iterative policy evaluation for the gridworld (4×4, no obstacles, terminal states at top-left and bottom-right, rewards -1 per step). Use a uniform random policy and \\(\gamma=1\\). Stop when the maximum change is less than 1e-4.

**Professor's hints**

- Represent the grid as a 4×4 array; terminal states (0,0) and (3,3) typically have value 0 (no further reward). Do not update \\(V\\) for terminal states in the loop, or set them to 0 and keep them fixed.
- Uniform random policy: from each non-terminal state, each of the 4 actions (up, down, left, right) has probability 0.25. If an action would go off the grid, the agent stays in place (and gets -1). So from (0,0) you do not update (it's terminal); from (0,1) "left" stays (0,1) with -1.
- Update: for each non-terminal state \\(s\\), compute the new value as the Bellman expectation (average over 4 actions of immediate reward + \\(\\gamma\\) × value of next state). Use synchronous updates: compute all new values from the *old* \\(V\\), then replace \\(V\\) with the new values.
- Stop when \\(\\max_s |V_{new}(s) - V_{old}(s)| < 10^{-4}\\).

**Common pitfalls**

- **Updating terminal states:** Terminal states should have value 0 (or by definition no update). Do not apply the Bellman update to (0,0) and (3,3); leave them at 0.
- **Asynchronous vs synchronous:** Synchronous = use the same \\(V\\) for all updates in one sweep. If you update in place (state by state), later states in the same sweep see already-updated values—that is asynchronous and can change convergence; stick to synchronous for this exercise.
- **Wrong transition model:** From a corner, two actions might hit the wall (stay, -1) and two might move. Count correctly; the probability of each next state depends on how many actions lead there.

{{< collapse summary="Worked solution (warm-up: 1×3 line)" >}}
**Warm-up:** For a 1×3 line (states 0, 1, 2), terminal at 0 and 2 with value 0, reward -1 per step, one action "move left" from 1 to 0, one action "move right" from 1 to 2. Compute \\(V(1)\\) by hand for \\(\gamma=1\\).

**Step 1:** From state 1 we take one action and reach either state 0 or state 2 (each terminal with value 0). So we get one step of reward -1 and then 0. Thus \\(V(1) = -1 + \gamma \cdot 0 = -1\\) (for \\(\gamma=1\\) or any \\(\gamma\\): \\(V(1) = -1\\)).

**Explanation:** One step to either terminal gives immediate reward -1 and then value 0. The same Bellman expectation structure is used in the 4×4 gridworld: each state’s value is immediate reward plus \\(\gamma\\) times the value of the next state, averaged over the policy.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For a 1×3 line (states 0, 1, 2), terminal at 0 and 2 with value 0, reward -1 per step, one action "move left" from 1 to 0, one action "move right" from 1 to 2. Compute \\(V(1)\\) by hand for \\(\\gamma=1\\) (one step to either terminal ⇒ -1).
2. **Coding:** Run policy evaluation (iterative Bellman expectation) on a 4×4 gridworld until convergence. Plot \\(V(s)\\) as a heatmap for a random policy (e.g. uniform over actions).
3. **Challenge:** After convergence, derive the **greedy policy** with respect to your \\(V\\): in each state, which action(s) maximize immediate reward + \\(V(s')\\)? Plot or print the greedy policy as arrows. Compare to the optimal policy (shortest path to nearest terminal).
4. **Variant:** Run policy evaluation with \\(\gamma=0.5\\) and \\(\gamma=1.0\\) on the same 4×4 gridworld. How does the value at the center cell differ between the two? Why is \\(\gamma=1\\) acceptable here (finite episode)?
5. **Debug:** The update below applies the Bellman update to terminal states (a common mistake), causing incorrect values. Find and fix it.

{{< pyrepl code="# 1x3 gridworld: states 0,1,2 — terminals at 0 and 2\nV = [0.0, 0.0, 0.0]\nfor sweep in range(20):\n    new_V = V[:]\n    for s in range(3):  # BUG: should skip terminals 0 and 2\n        if s == 1:\n            # state 1: left goes to 0, right goes to 2, reward -1 each\n            new_V[s] = 0.5*(-1 + V[0]) + 0.5*(-1 + V[2])\n        else:\n            new_V[s] = -1 + V[s]  # BUG: terminal updated\n    V = new_V\nprint('V:', V)  # should be [0, -1, 0]" height="240" >}}

6. **Conceptual:** Why must we use synchronous (not in-place) updates when running one full sweep of policy evaluation? What can go wrong with in-place updates?
7. **Recall:** State the iterative policy evaluation update rule \\(V(s) \\leftarrow \\ldots\\) from memory.
