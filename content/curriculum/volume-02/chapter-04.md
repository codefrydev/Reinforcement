---
title: "Chapter 14: Q-Learning (Off-Policy TD Control)"
description: "Q-learning on Cliff Walking; compare with SARSA."
date: 2026-03-10T00:00:00Z
weight: 14
draft: false
---

**Learning objectives**

- Implement Q-learning: update \\(Q(s,a)\\) using target \\(r + \\gamma \\max_{a'} Q(s',a')\\) (off-policy).
- Compare Q-learning and SARSA on Cliff Walking: paths and reward curves.
- Explain why Q-learning can learn a riskier policy (cliff edge) than SARSA.

**Concept and real-world RL**

**Q-learning** is off-policy: it updates \\(Q(s,a)\\) using the *greedy* next action (\\(\\max_{a'} Q(s',a')\\)), so it learns the value of the optimal policy while you can behave with \\(\epsilon\\)-greedy (or any exploration). The update is \\(Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]\\). On Cliff Walking, Q-learning often converges to the *shortest* path along the cliff (high reward when no exploration, but dangerous if you occasionally take a random step). SARSA learns the *actual* policy including exploration and tends to stay away from the cliff. In practice, Q-learning is simple and widely used (e.g. DQN); when safety matters, on-policy or conservative methods may be preferred.

**Exercise:** Implement Q-learning for the same Cliff Walking environment. Compare the learned paths and total rewards with SARSA. Explain why Q-learning might prefer the cliff edge while SARSA takes a safer path.

**Professor's hints**

- Same setup as SARSA (Cliff Walking, \\(\epsilon\\)-greedy for behavior). The only change: when updating \\(Q(s,a)\\), use target \\(r + \\gamma \\max_{a'} Q(s',a')\\), not \\(r + \\gamma Q(s',a')\\). You still *choose* the next action with \\(\epsilon\\)-greedy for the next step; you just use *max* in the update.
- To compare paths: after training, run a few episodes with \\(\epsilon=0\\) (greedy) and record the states visited. Visualize or print the path. Q-learning's greedy path often walks along the cliff; SARSA's often one row up.
- Explanation: Q-learning assumes the agent will act greedily in the future, so it values states by the best possible outcome. SARSA values states by what *will* happen when the agent sometimes explores, so it penalizes being near the cliff (where a random step is costly).

**Common pitfalls**

- **Using a' in the target:** If you use \\(Q(s',a')\\) with the actual \\(a'\\) you are doing SARSA. Q-learning must use \\(\\max_{a'} Q(s',a')\\).
- **Behavior policy:** You still need to explore (e.g. \\(\epsilon\\)-greedy) to visit all state-action pairs. The *target* policy is greedy; the *behavior* policy is \\(\epsilon\\)-greedy.
- **Comparing with same epsilon:** Use the same \\(\epsilon\\) for both algorithms when comparing. When evaluating (plotting paths), use \\(\epsilon=0\\) so you see the learned greedy policy.

**Extra practice**

1. **Warm-up:** Write the Q-learning update in one line. What is the TD target? How does it differ from SARSA's target?
2. **Challenge:** Run Q-learning with \\(\epsilon=0.1\\) for many episodes, then run 100 evaluation episodes with \\(\epsilon=0\\). Also run 100 evaluation episodes with \\(\epsilon=0.1\\) (so the agent sometimes steps off the cliff). Compare average reward: greedy evaluation vs behavioral evaluation. Why does the latter get worse?
