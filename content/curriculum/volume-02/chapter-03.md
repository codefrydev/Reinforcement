---
title: "Chapter 13: SARSA (On-Policy TD Control)"
description: "SARSA on Cliff Walking; plot sum of rewards per episode."
date: 2026-03-10T00:00:00Z
weight: 13
draft: false
tags: ["SARSA", "on-policy", "Cliff Walking", "curriculum"]
keywords: ["SARSA", "on-policy TD control", "Cliff Walking", "temporal difference"]
---

**Learning objectives**

- Implement SARSA: update \\(Q(s,a)\\) using the transition \\((s,a,r,s',a')\\) with target \\(r + \\gamma Q(s',a')\\).
- Use \\(\epsilon\\)-greedy exploration for behavior and learn the same policy you follow (on-policy).
- Interpret learning curves (sum of rewards per episode) on Cliff Walking.

**Concept and real-world RL**

**SARSA** is an on-policy TD control method: it updates \\(Q(s,a)\\) using the actual next action \\(a'\\) chosen by the current policy, so it learns the value of the *behavior* policy (the one you are following). The update is \\(Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma Q(s',a') - Q(s,a)]\\). Because \\(a'\\) can be exploratory, SARSA accounts for the risk of exploration (e.g. stepping off the cliff by accident) and often learns a safer policy than Q-learning on Cliff Walking. In real applications, on-policy methods are used when you want to optimize the same policy you use for data collection (e.g. safe robotics).

**Illustration (SARSA learning curve):** On Cliff Walking, the sum of rewards per episode typically improves as \\(Q\\) is learned, then stabilizes. The chart below shows a typical episode-reward curve over training.

{{< chart type="line" palette="return" title="Sum of rewards per episode (SARSA on Cliff Walking)" labels="0, 100, 200, 300, 400, 500" data="-80, -45, -25, -15, -13, -12" xLabel="Episode" yLabel="Return" >}}

**Exercise:** Implement SARSA to learn an optimal policy for the Cliff Walking environment (from Sutton & Barto). Use \\(\epsilon\\)-greedy exploration with \\(\epsilon=0.1\\), \\(\alpha=0.5\\), \\(\gamma=0.9\\). Plot the sum of rewards per episode.

**Professor's hints**

- Cliff Walking: grid with a cliff along one row; stepping into the cliff gives a large negative reward and resets to start. Goal is to reach the end. Use Gymnasium's "CliffWalking-v0" or implement a simple grid.
- Maintain \\(Q(s,a)\\) as a dict or 2D array. Each step: observe \\(s, a, r, s'\\), choose \\(a'\\) from \\(s'\\) with \\(\epsilon\\)-greedy, then update \\(Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma Q(s',a') - Q(s,a)]\\). Use \\(a'\\) (the action you will take next), not max over actions.
- Plot: x = episode index, y = total reward in that episode. You should see improvement over episodes and possibly some variance.

**Common pitfalls**

- **Using max instead of a':** SARSA uses \\(Q(s',a')\\) where \\(a'\\) is the action *actually taken* in \\(s'\\). If you use \\(\\max_{a'} Q(s',a')\\) you have Q-learning, not SARSA.
- **When to choose a':** Choose \\(a'\\) *after* arriving in \\(s'\\), before the next env.step. So the loop is: step → get \\(s', r\\) → choose \\(a'\\) from \\(s'\\) → update \\(Q(s,a)\\) with \\(Q(s',a')\\) → set \\(s,a = s',a'\\) and repeat.
- **Terminal state:** When \\(s'\\) is terminal, \\(Q(s',a') = 0\\) (no next step). So the target is just \\(r\\) for the last transition.

{{< collapse summary="Worked solution (warm-up: SARSA update and TD error)" >}}
**Warm-up:** For one transition \\((s,a,r,s',a')\\), write the SARSA update in one line. What is the TD error? **Answer:** \\(Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma Q(s',a') - Q(s,a)]\\). The TD error is \\(\\delta = r + \\gamma Q(s',a') - Q(s,a)\\). We use the *actual* next action \\(a'\\) (on-policy); Q-learning would use \\(\\max_{a'} Q(s',a')\\) instead.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For one transition \\((s,a,r,s',a')\\), write the SARSA update in one line. What is the TD error \\(r + \\gamma Q(s',a') - Q(s,a)\\)?
2. **Coding:** Implement SARSA for a 5×5 gridworld with a goal. Use a tabular Q-table, ε-greedy (ε=0.1). Run 500 episodes and plot episode return vs episode.
3. **Challenge:** Run SARSA with \\(\epsilon=0\\) (greedy) from the start. Does it learn a good policy? Compare with \\(\epsilon=0.1\\). Why does some exploration help?
4. **Variant:** Run SARSA with \\(\epsilon=0.05\\) and \\(\epsilon=0.3\\) on Cliff Walking. Compare episode return curves and the final policy. Which \\(\epsilon\\) leads to a safer path?
5. **Debug:** The code below uses \\(\\max_{a'} Q(s',a')\\) instead of \\(Q(s',a')\\) where \\(a'\\) is the action actually taken. This makes it Q-learning, not SARSA. Fix it.

{{< pyrepl code="import random\nQ = {}  # defaultdict-like; initialize lazily\nactions = [0,1,2,3]\nalpha, gamma, eps = 0.5, 0.9, 0.1\n\ndef get_q(s, a):\n    return Q.get((s,a), 0.0)\n\ndef sarsa_update(s, a, r, s_next, a_next):\n    # BUG: uses max instead of Q(s', a_next)\n    td_target = r + gamma * max(get_q(s_next, a2) for a2 in actions)\n    Q[(s,a)] = get_q(s,a) + alpha * (td_target - get_q(s,a))\n\n# Fix: use get_q(s_next, a_next) — the actual next action\nsarsa_update('S0', 0, -1, 'S1', 1)\nprint('Q[(S0,0)]:', Q.get(('S0',0), 0))" height="240" >}}

6. **Conceptual:** Why does SARSA learn a safer policy than Q-learning on Cliff Walking, even when both use ε-greedy exploration?
7. **Recall:** Write the SARSA update rule from memory in one line.
