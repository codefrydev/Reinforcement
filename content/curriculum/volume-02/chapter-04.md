---
title: "Chapter 14: Q-Learning (Off-Policy TD Control)"
description: "Q-learning on Cliff Walking; compare with SARSA."
date: 2026-03-10T00:00:00Z
weight: 14
draft: false
difficulty: 6
tags: ["Q-learning", "off-policy", "Cliff Walking", "curriculum"]
keywords: ["Q-learning", "off-policy", "Cliff Walking", "TD control"]
roadmap_color: "teal"
roadmap_icon: "book"
roadmap_phase_label: "Vol 2 · Ch 4"
---

{{< notebook path="volume-02/ch14_q_learning.ipynb" title="Open Q-learning notebook" >}}

**Learning objectives**

- Implement Q-learning: update \\(Q(s,a)\\) using target \\(r + \\gamma \\max_{a'} Q(s',a')\\) (off-policy).
- Compare Q-learning and SARSA on Cliff Walking: paths and reward curves.
- Explain why Q-learning can learn a riskier policy (cliff edge) than SARSA.

**Concept and real-world RL**

**Q-learning** is off-policy: it updates \\(Q(s,a)\\) using the *greedy* next action (\\(\\max_{a'} Q(s',a')\\)), so it learns the value of the optimal policy while you can behave with \\(\epsilon\\)-greedy (or any exploration). The update is \\(Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]\\). On Cliff Walking, Q-learning often converges to the *shortest* path along the cliff (high reward when no exploration, but dangerous if you occasionally take a random step). SARSA learns the *actual* policy including exploration and tends to stay away from the cliff. In practice, Q-learning is simple and widely used (e.g. DQN); when safety matters, on-policy or conservative methods may be preferred.

**Illustration (Q-learning vs SARSA):** When evaluated greedily, Q-learning often achieves higher mean reward (short path) while SARSA is more conservative. The chart below compares typical average episode return after training (greedy evaluation).

{{< chart type="bar" palette="comparison" title="Mean episode return (greedy eval, Cliff Walking)" labels="Q-learning, SARSA" data="-13, -17" yLabel="Mean return" >}}

**Exercise:** Implement Q-learning for the same Cliff Walking environment. Compare the learned paths and total rewards with SARSA. Explain why Q-learning might prefer the cliff edge while SARSA takes a safer path.

{{< pyrepl code="# Q-learning update step\nQ_sa = 0.3   # current Q(s,a)\nr = 1        # reward\ngamma = 0.9\nalpha = 0.1\n# max Q(s', a') for next state\nmax_q_next = 0.5\n\n# TODO: target = r + gamma * max_q_next\n# Q_sa_new = Q_sa + alpha * (target - Q_sa)\ntarget = None\nQ_sa_new = None\nprint(f'target = {target}')\nprint(f'Q new = {Q_sa_new}')  # expected: 0.327" height="240" >}}

**Professor's hints**

- Same setup as SARSA (Cliff Walking, \\(\epsilon\\)-greedy for behavior). The only change: when updating \\(Q(s,a)\\), use target \\(r + \\gamma \\max_{a'} Q(s',a')\\), not \\(r + \\gamma Q(s',a')\\). You still *choose* the next action with \\(\epsilon\\)-greedy for the next step; you just use *max* in the update.
- To compare paths: after training, run a few episodes with \\(\epsilon=0\\) (greedy) and record the states visited. Visualize or print the path. Q-learning's greedy path often walks along the cliff; SARSA's often one row up.
### Explanation

Q-learning assumes the agent will act greedily in the future, so it values states by the best possible outcome. SARSA values states by what *will* happen when the agent sometimes explores, so it penalizes being near the cliff (where a random step is costly).

**Common pitfalls**

- **Using a' in the target:** If you use \\(Q(s',a')\\) with the actual \\(a'\\) you are doing SARSA. Q-learning must use \\(\\max_{a'} Q(s',a')\\).
- **Behavior policy:** You still need to explore (e.g. \\(\epsilon\\)-greedy) to visit all state-action pairs. The *target* policy is greedy; the *behavior* policy is \\(\epsilon\\)-greedy.
- **Comparing with same epsilon:** Use the same \\(\epsilon\\) for both algorithms when comparing. When evaluating (plotting paths), use \\(\epsilon=0\\) so you see the learned greedy policy.

{{< collapse summary="Worked solution (warm-up: Q-learning update vs SARSA)" >}}
**Warm-up:** Write the Q-learning update in one line. What is the TD target? How does it differ from SARSA? **Answer:** \\(Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]\\). The TD target is \\(r + \\gamma \\max_{a'} Q(s',a')\\). SARSA uses \\(r + \\gamma Q(s',a')\\) with the *actual* next action \\(a'\\); Q-learning uses the *max* over next actions, so it learns the optimal Q while you can still behave with \\(\\epsilon\\)-greedy.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Write the Q-learning update in one line. What is the TD target? How does it differ from SARSA's target?
2. **Coding:** Implement Q-learning on the same 5×5 gridworld as in the SARSA coding exercise. Compare the learned Q with SARSA after 500 episodes (e.g. max difference in Q-values).
3. **Challenge:** Run Q-learning with \\(\epsilon=0.1\\) for many episodes, then run 100 evaluation episodes with \\(\epsilon=0\\). Also run 100 evaluation episodes with \\(\epsilon=0.1\\) (so the agent sometimes steps off the cliff). Compare average reward: greedy evaluation vs behavioral evaluation. Why does the latter get worse?
4. **Variant:** Run Q-learning with \\(\alpha=0.1\\) and \\(\alpha=0.9\\) on Cliff Walking. Which converges faster? Does high \\(\alpha\\) cause instability?
5. **Debug:** The update below uses \\(Q(s',a')\\) where \\(a'\\) is the action actually selected—making it SARSA, not Q-learning. Fix it to use \\(\\max_{a'} Q(s',a')\\).

{{< pyrepl code="Q = {}\nactions = [0,1,2,3]\nalpha, gamma, eps = 0.1, 0.9, 0.1\n\ndef get_q(s, a): return Q.get((s,a), 0.0)\n\ndef q_learning_update(s, a, r, s_next, a_next):\n    # BUG: uses a_next (SARSA) instead of max over actions\n    td_target = r + gamma * get_q(s_next, a_next)\n    Q[(s,a)] = get_q(s,a) + alpha * (td_target - get_q(s,a))\n\n# Fix: replace get_q(s_next, a_next) with max Q(s_next, a') over all a'\nq_learning_update('S0', 0, -1, 'S1', 1)\nprint('Q[(S0,0)]:', Q.get(('S0',0), 0))" height="220" >}}

6. **Conceptual:** Q-learning is off-policy. What does "off-policy" mean, and why does it allow Q-learning to learn the optimal policy while using ε-greedy behavior?
7. **Recall:** Write the Q-learning update rule from memory.
