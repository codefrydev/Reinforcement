---
title: "Chapter 8: Dynamic Programming — Policy Iteration"
description: "Policy iteration and comparison with value iteration."
date: 2026-03-10T00:00:00Z
weight: 8
draft: false
tags: ["dynamic programming", "policy iteration", "value iteration", "curriculum"]
keywords: ["policy iteration", "value iteration", "dynamic programming", "optimal policy"]
---

**Learning objectives**

- Implement policy iteration: alternate policy evaluation and greedy policy improvement.
- Recognize that the policy stabilizes in a finite number of iterations for finite MDPs.
- Compare the resulting policy and value function with value iteration.

**Concept and real-world RL**

**Policy iteration** alternates two steps: (1) **policy evaluation**—compute \\(V^\\pi\\) for the current policy \\(\\pi\\); (2) **policy improvement**—update \\(\\pi\\) to be greedy with respect to \\(V^\\pi\\). The new policy is at least as good as the old (and strictly better unless already optimal). Repeating this process converges to the optimal policy in a finite number of iterations (for finite MDPs). It is a cornerstone of dynamic programming for RL; in practice, we often do only a few evaluation steps (generalized policy iteration) or use value iteration, which interleaves evaluation and improvement in one update.

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
