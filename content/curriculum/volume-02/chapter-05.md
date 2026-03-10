---
title: "Chapter 15: Expected SARSA"
description: "Expected SARSA vs Q-learning; variance and learning curves."
date: 2026-03-10T00:00:00Z
weight: 15
draft: false
---

**Learning objectives**

- Implement Expected SARSA: use \\(\\sum_{a'} \\pi(a'|s') Q(s',a')\\) as the target instead of \\(\\max_{a'} Q(s',a')\\) or \\(Q(s',a')\\).
- Relate Expected SARSA to SARSA (on-policy) and Q-learning (max); it can be used on- or off-policy depending on \\(\\pi\\).
- Compare update variance and learning curves with Q-learning.

**Concept and real-world RL**

**Expected SARSA** uses the *expected* next action value under a policy \\(\\pi\\): target = \\(r + \\gamma \\sum_{a'} \\pi(a'|s') Q(s',a')\\). For \\(\epsilon\\)-greedy \\(\\pi\\), this is \\(r + \\gamma [(1-\\epsilon) \\max_{a'} Q(s',a') + \\epsilon \\cdot \\text{(uniform over actions)}]\\). It reduces the variance of the update (compared to SARSA, which uses a single sample \\(Q(s',a')\\)) and can be more stable. When \\(\\pi\\) is greedy, Expected SARSA becomes Q-learning. In practice, it is a middle ground between SARSA and Q-learning and is used in some deep RL variants.

**Exercise:** Modify your Q-learning code to implement Expected SARSA. Instead of using the max over next actions, use the expected value under the current \\(\epsilon\\)-greedy policy. Compare the variance of the updates and the learning curves.

**Professor's hints**

- For \\(\epsilon\\)-greedy with \\(n\\) actions: \\(\\sum_{a'} \\pi(a'|s') Q(s',a') = (1-\\epsilon) \\max_{a'} Q(s',a') + \\epsilon \\cdot \\frac{1}{n} \\sum_{a'} Q(s',a')\\). Or: with probability \\(1-\\epsilon\\) the greedy action, with probability \\(\\epsilon\\) uniform; so expected value = \\((1-\\epsilon) Q(s',a^*) + \\epsilon \\cdot \\frac{1}{n} \\sum_{a'} Q(s',a')\\).
- Implement: compute \\(\\text{target} = r + \\gamma \\cdot \\text{expected_value}\\), then \\(Q(s,a) \\leftarrow Q(s,a) + \\alpha (\\text{target} - Q(s,a))\\).
- Variance: you can track the squared TD error over many steps and compare Expected SARSA vs Q-learning. Expected SARSA typically has lower variance because the target does not depend on a random \\(a'\\). Learning curves: plot average reward per episode for both; Expected SARSA may converge more smoothly.

**Common pitfalls**

- **Wrong formula for expected value:** The expected value under \\(\epsilon\\)-greedy is *not* just \\(\\max_a Q(s',a)\\). It is a weighted average of the greedy value and the average of all Q-values.
- **Using the same \\(\\pi\\) for behavior and target:** For Expected SARSA we usually use the same \\(\epsilon\\)-greedy policy for both (on-policy). So the target uses the *current* policy's expectation, and we collect data with that policy.
- **Numerical precision:** For many actions, \\(\\frac{1}{n}\\sum_a Q(s',a)\\) can be computed once; avoid recomputing in a loop if \\(n\\) is large.

**Extra practice**

1. **Warm-up:** For 2 actions and \\(\epsilon=0.1\\), write the Expected SARSA target in terms of \\(Q(s',0)\\) and \\(Q(s',1)\\) when the greedy action is 0.
2. **Coding:** Implement Expected SARSA (target = expected Q under ε-greedy policy) for gridworld. Compare learning curve with Q-learning over 500 episodes.
3. **Challenge:** Implement Expected SARSA where the *target* policy is greedy (\\(\\pi(a'|s') = 1\\) for \\(a' = \\arg\\max\\)) but the *behavior* policy is \\(\epsilon\\)-greedy. How does this differ from Q-learning? (It should be identical in the limit; the update is the same when the target policy is greedy.)
