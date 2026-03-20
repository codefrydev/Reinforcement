---
title: "Chapter 15: Expected SARSA"
description: "Expected SARSA vs Q-learning; variance and learning curves."
date: 2026-03-10T00:00:00Z
weight: 15
draft: false
difficulty: 6
tags: ["Expected SARSA", "Q-learning", "variance", "curriculum"]
keywords: ["Expected SARSA", "Q-learning", "variance", "learning curves"]
roadmap_color: "teal"
roadmap_icon: "book"
roadmap_phase_label: "Vol 2 · Ch 5"
---

**Learning objectives**

- Implement Expected SARSA: use \\(\\sum_{a'} \\pi(a'|s') Q(s',a')\\) as the target instead of \\(\\max_{a'} Q(s',a')\\) or \\(Q(s',a')\\).
- Relate Expected SARSA to SARSA (on-policy) and Q-learning (max); it can be used on- or off-policy depending on \\(\\pi\\).
- Compare update variance and learning curves with Q-learning.

**Concept and real-world RL**

**Expected SARSA** uses the *expected* next action value under a policy \\(\\pi\\): target = \\(r + \\gamma \\sum_{a'} \\pi(a'|s') Q(s',a')\\). For \\(\epsilon\\)-greedy \\(\\pi\\), this is \\(r + \\gamma [(1-\\epsilon) \\max_{a'} Q(s',a') + \\epsilon \\cdot \\text{(uniform over actions)}]\\). It reduces the variance of the update (compared to SARSA, which uses a single sample \\(Q(s',a')\\)) and can be more stable. When \\(\\pi\\) is greedy, Expected SARSA becomes Q-learning. In practice, it is a middle ground between SARSA and Q-learning and is used in some deep RL variants.

**Illustration (update variance):** Expected SARSA uses the expectation over the next action instead of a single sample, so the TD target has lower variance. The chart below compares typical squared TD error (averaged over many steps) for Expected SARSA vs Q-learning.

{{< chart type="bar" palette="comparison" title="Mean squared TD error (lower is more stable)" labels="Expected SARSA, Q-learning" data="0.08, 0.15" yLabel="TD error" >}}

**Exercise:** Modify your Q-learning code to implement Expected SARSA. Instead of using the max over next actions, use the expected value under the current \\(\epsilon\\)-greedy policy. Compare the variance of the updates and the learning curves.

**Professor's hints**

- For \\(\epsilon\\)-greedy with \\(n\\) actions: \\(\\sum_{a'} \\pi(a'|s') Q(s',a') = (1-\\epsilon) \\max_{a'} Q(s',a') + \\epsilon \\cdot \\frac{1}{n} \\sum_{a'} Q(s',a')\\). Or: with probability \\(1-\\epsilon\\) the greedy action, with probability \\(\\epsilon\\) uniform; so expected value = \\((1-\\epsilon) Q(s',a^*) + \\epsilon \\cdot \\frac{1}{n} \\sum_{a'} Q(s',a')\\).
- Implement: compute \\(\\text{target} = r + \\gamma \\cdot \\text{expected_value}\\), then \\(Q(s,a) \\leftarrow Q(s,a) + \\alpha (\\text{target} - Q(s,a))\\).
- Variance: you can track the squared TD error over many steps and compare Expected SARSA vs Q-learning. Expected SARSA typically has lower variance because the target does not depend on a random \\(a'\\). Learning curves: plot average reward per episode for both; Expected SARSA may converge more smoothly.

**Common pitfalls**

- **Wrong formula for expected value:** The expected value under \\(\epsilon\\)-greedy is *not* just \\(\\max_a Q(s',a)\\). It is a weighted average of the greedy value and the average of all Q-values.
- **Using the same \\(\\pi\\) for behavior and target:** For Expected SARSA we usually use the same \\(\epsilon\\)-greedy policy for both (on-policy). So the target uses the *current* policy's expectation, and we collect data with that policy.
- **Numerical precision:** For many actions, \\(\\frac{1}{n}\\sum_a Q(s',a)\\) can be computed once; avoid recomputing in a loop if \\(n\\) is large.

{{< collapse summary="Worked solution (warm-up: Expected SARSA target for 2 actions)" >}}
**Warm-up:** For 2 actions and \\(\\epsilon=0.1\\), write the Expected SARSA target in terms of \\(Q(s',0)\\) and \\(Q(s',1)\\) when the greedy action is 0. **Step 1:** Under \\(\\epsilon\\)-greedy, with probability \\(1-\\epsilon\\) we take action 0, with probability \\(\\epsilon\\) we take action 0 or 1 uniformly (so 0.5 each). So expected next value = \\((1-0.1) Q(s',0) + 0.1 \\cdot \\frac{1}{2}[Q(s',0)+Q(s',1)] = 0.9 Q(s',0) + 0.05 Q(s',0) + 0.05 Q(s',1) = 0.95 Q(s',0) + 0.05 Q(s',1)\\). **Step 2:** Target = \\(r + \\gamma (0.95 Q(s',0) + 0.05 Q(s',1))\\). This is the Expected SARSA update; it uses the expectation over the next action instead of a single sample (SARSA) or max (Q-learning).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For 2 actions and \\(\epsilon=0.1\\), write the Expected SARSA target in terms of \\(Q(s',0)\\) and \\(Q(s',1)\\) when the greedy action is 0.
2. **Coding:** Implement Expected SARSA (target = expected Q under ε-greedy policy) for gridworld. Compare learning curve with Q-learning over 500 episodes.
3. **Challenge:** Implement Expected SARSA where the *target* policy is greedy (\\(\\pi(a'|s') = 1\\) for \\(a' = \\arg\\max\\)) but the *behavior* policy is \\(\epsilon\\)-greedy. How does this differ from Q-learning? (It should be identical in the limit; the update is the same when the target policy is greedy.)
4. **Variant:** Try \\(\epsilon=0.0\\) (greedy target) and \\(\epsilon=0.3\\) in Expected SARSA. For \\(\epsilon=0\\), is the target the same as Q-learning? Verify numerically.
5. **Debug:** The expected-value computation below uses \\(\\max Q(s',a')\\) instead of the weighted average under ε-greedy. Fix it.

{{< pyrepl code="Q_next = [0.3, 0.8, 0.1]  # Q(s', a) for 3 actions\neps = 0.1\nn_actions = len(Q_next)\n\n# BUG: uses max instead of expected value under eps-greedy\ndef expected_q_buggy(Q_next, eps):\n    return max(Q_next)\n\n# Fix: use (1-eps)*max + eps*(mean)\ndef expected_q_correct(Q_next, eps):\n    greedy_val = max(Q_next)\n    mean_val = sum(Q_next) / len(Q_next)\n    return (1 - eps) * greedy_val + eps * mean_val\n\nprint('Buggy:', expected_q_buggy(Q_next, eps))\nprint('Correct:', expected_q_correct(Q_next, eps))" height="240" >}}

6. **Conceptual:** When does Expected SARSA have lower variance than (sample-based) SARSA? Why does taking the expectation reduce variance?
7. **Recall:** Write the Expected SARSA target for an ε-greedy behavior policy with \\(n\\) actions from memory.
