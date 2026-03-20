---
title: "Chapter 8: Dynamic Programming — Policy Iteration"
description: "Policy iteration and comparison with value iteration."
date: 2026-03-10T00:00:00Z
weight: 8
draft: false
difficulty: 6
tags: ["dynamic programming", "policy iteration", "value iteration", "curriculum"]
keywords: ["policy iteration", "value iteration", "dynamic programming", "optimal policy"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Ch 8"
---

**Learning objectives**

- Implement policy iteration: alternate policy evaluation and greedy policy improvement.
- Recognize that the policy stabilizes in a finite number of iterations for finite MDPs.
- Compare the resulting policy and value function with value iteration.

**Concept and real-world RL**

**Policy iteration** alternates two steps: (1) **policy evaluation**—compute \\(V^\\pi\\) for the current policy \\(\\pi\\); (2) **policy improvement**—update \\(\\pi\\) to be greedy with respect to \\(V^\\pi\\). The new policy is at least as good as the old (and strictly better unless already optimal). Repeating this process converges to the optimal policy in a finite number of iterations (for finite MDPs). It is a cornerstone of dynamic programming for RL; in practice, we often do only a few evaluation steps (generalized policy iteration) or use value iteration, which interleaves evaluation and improvement in one update.

**Illustration (policy improvement):** In each improvement step, we count how many states change their action. Typically this number drops over iterations until no state changes. The chart below shows an example: states changed per improvement step.

{{< chart type="bar" title="States with changed action per improvement step" labels="Step 1, Step 2, Step 3, Step 4" data="14, 6, 2, 0" >}}

**Exercise:** Extend your code from Chapter 7 to perform policy iteration. After evaluating the policy, improve it greedily with respect to the current value function. Repeat until the policy stabilizes. Compare the final policy with value iteration results.

**Professor's hints**

- Reuse your iterative policy evaluation from Chapter 7. Run it until convergence (or a fixed number of sweeps) to get \\(V\\) for the current policy.
- Greedy improvement: for each state \\(s\\), set \\(\\pi(s) = \\arg\\max_a \\sum_{s',r} P(s',r|s,a)[r + \\gamma V(s')]\\). For the gridworld, compute the one-step lookahead for each action (reward -1 plus \\(\\gamma\\) × value of next state); choose the action that maximizes this. Handle ties (e.g. pick first).
- Stop when the policy does not change in a full sweep. Compare: run value iteration (Chapter 9) on the same gridworld; the optimal \\(V\\) and greedy policy should match policy iteration's final result.

**Common pitfalls**

- **Improving before evaluation converges:** The theory assumes we evaluate until \\(V = V^\\pi\\) (or close). If you do only one evaluation sweep per iteration, you are doing "value iteration-like" updates; it still works but is not strict policy iteration.
- **Ties in argmax:** If two actions have the same one-step value, pick one consistently (e.g. lowest index). The policy is still optimal; only the representation of the optimal policy may differ.
- **Comparing policies:** Policy iteration and value iteration both yield the same optimal \\(V^*\\) and an optimal policy. Differences in the *reported* policy can occur only due to tie-breaking or implementation bugs.

{{< collapse summary="Worked solution (warm-up: one round of policy iteration)" >}}
**Warm-up:** After one round of policy iteration (evaluate once, improve once), is the new policy necessarily different from the initial random policy? Explain in one sentence.

**Answer:** Not necessarily. The new policy is greedy with respect to \\(V^\pi\\) for the *current* policy \\(\pi\\). If the initial random policy happened to be already greedy with respect to its own value function (e.g. by chance every state’s best one-step action was chosen with positive probability), the improved policy could be the same. In practice, for a 4×4 gridworld with uniform random, the first improvement usually *does* change the policy (e.g. states near a terminal become greedy toward that terminal). So one round gives a *possibly* different policy; it is at least as good as the initial one.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** After one round of policy iteration (evaluate once, improve once), is the new policy necessarily different from the initial random policy? Explain in one sentence.
2. **Coding:** Implement policy iteration (evaluate until convergence, then improve, repeat). On a 4×4 gridworld, start with a random policy and run until the policy does not change. Compare the final \\(V\\) with value iteration.
3. **Challenge:** Count how many policy iteration iterations (evaluate + improve) you need until the policy stabilizes for the 4×4 gridworld. Compare with the number of value iteration sweeps needed to reach the same \\(V^*\\) (within 1e-4). Which converged faster in your implementation?
4. **Variant:** Instead of running policy evaluation until full convergence each iteration, perform exactly 5 Bellman sweeps then improve. Does the final policy change? How many improvement steps are needed compared to full evaluation?
5. **Debug:** The "policy stable" check below never triggers because it compares lists by identity instead of value. Find and fix the bug.

{{< pyrepl code="import copy\npolicy = [[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]]  # 4x4 grid\n\ndef improve(policy, V):\n    new_policy = copy.deepcopy(policy)\n    # ... (update new_policy based on V)\n    return new_policy\n\ndef policy_iteration(policy, V):\n    for _ in range(100):\n        new_policy = improve(policy, V)\n        if new_policy is policy:  # BUG: identity check, not equality\n            print('converged')\n            break\n        policy = new_policy\n    return policy\n\n# Fix: use == or compare elementwise\nprint('Policy stable check: use == not is')" height="220" >}}

6. **Conceptual:** Why does policy improvement guarantee the new policy is at least as good as the old one? State the key inequality that makes this work.
7. **Recall:** State the policy improvement theorem in one sentence from memory.
