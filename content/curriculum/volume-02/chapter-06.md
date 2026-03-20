---
title: "Chapter 16: N-Step Bootstrapping"
description: "n-step SARSA (n=4) on Cliff Walking."
date: 2026-03-10T00:00:00Z
weight: 16
draft: false
difficulty: 6
tags: ["n-step", "SARSA", "Cliff Walking", "bootstrapping", "curriculum"]
keywords: ["n-step SARSA", "n-step bootstrapping", "Cliff Walking", "TD"]
roadmap_color: "teal"
roadmap_icon: "book"
roadmap_phase_label: "Vol 2 · Ch 6"
---

**Learning objectives**

- Implement n-step SARSA: accumulate \\(n\\) steps of experience, then update \\(Q(s_0,a_0)\\) using the n-step return \\(r_1 + \\gamma r_2 + \\cdots + \\gamma^{n-1} r_n + \\gamma^n Q(s_n,a_n)\\).
- Compare n-step (\\(n=4\\)) with one-step SARSA on Cliff Walking (learning speed, stability).
- Understand the trade-off: n-step uses more information per update but delays the update.

**Concept and real-world RL**

**N-step bootstrapping** uses a return over \\(n\\) steps: \\(G_{t:t+n} = r_{t+1} + \\gamma r_{t+2} + \\cdots + \\gamma^{n-1} r_{t+n} + \\gamma^n V(s_{t+n})\\) (or \\(Q(s_{t+n},a_{t+n})\\) for SARSA). \\(n=1\\) is TD(0); \\(n=\\infty\\) (until terminal) is Monte Carlo. Intermediate \\(n\\) balances bias and variance. In practice, n-step methods (e.g. n-step SARSA, A3C's n-step returns) can learn faster than one-step when \\(n\\) is chosen well; too large \\(n\\) delays updates and can hurt in non-stationary or long episodes.

**Illustration (n-step vs one-step):** For the same number of env steps, n-step SARSA often improves average reward faster early on because each update uses more reward signal. The chart below shows a typical comparison on Cliff Walking.

{{< chart type="line" palette="return" title="Mean reward per episode (same number of steps)" labels="0, 100, 200, 300, 400, 500" data="-60, -35, -22, -15, -13, -12" xLabel="Episode" yLabel="Mean return" >}}

**Exercise:** Implement n-step SARSA (with \\(n=4\\)) for the Cliff Walking environment. Write a function that accumulates n steps before updating. Compare the learning speed with one-step SARSA.

**Professor's hints**

- Store a buffer of the last \\(n\\) transitions \\((s_t, a_t, r_t)\\). When you have \\(n\\) steps, compute \\(G = r_1 + \\gamma r_2 + \\gamma^2 r_3 + \\gamma^3 r_4 + \\gamma^4 Q(s_4, a_4)\\) and update \\(Q(s_0, a_0) \\leftarrow Q(s_0, a_0) + \\alpha [G - Q(s_0, a_0)]\\). Then shift the buffer (drop the oldest, add the new \\(s,a\\) for the next step) or use a circular buffer.
- Handle episode end: if the episode terminates before \\(n\\) steps, use the truncated return (no bootstrap) or bootstrap with 0 for the terminal state.
- Comparison: run one-step SARSA and n-step SARSA for the same number of *steps* (or episodes). Plot average reward per episode. n-step often improves faster early on because each update uses more reward signal.

**Common pitfalls**

- **Indexing the buffer:** \\(s_0, a_0\\) is the state-action being updated; \\(r_1\\) is the reward after taking \\(a_0\\), and \\(s_n, a_n\\) is the state-action after \\(n\\) steps. Align indices carefully.
- **Updating only every n steps:** You update \\(Q(s_0,a_0)\\) when you have collected \\(n\\) steps. You can also do multi-step updates (update every \\((s_i, a_i)\\) in the buffer with its n-step return from that point); the exercise asks for the simpler "update the oldest when we have n steps."
- **Terminal state:** If \\(s_n\\) is terminal, there is no \\(a_n\\); use \\(G = r_1 + \\cdots + \\gamma^{n-1} r_n\\) (no \\(Q(s_n,a_n)\\) term, or \\(Q(\\text{terminal},\\cdot)=0\\)).

{{< collapse summary="Worked solution (warm-up: n=2 return)" >}}
**Warm-up:** For \\(n=2\\), write the n-step return \\(G_{t:t+2}\\) in terms of \\(r_{t+1}, r_{t+2}, Q(s_{t+2}, a_{t+2})\\) and \\(\\gamma\\). **Answer:** \\(G_{t:t+2} = r_{t+1} + \\gamma r_{t+2} + \\gamma^2 Q(s_{t+2}, a_{t+2})\\). So we take two rewards and then bootstrap with the Q-value at the state-action after 2 steps. For \\(n=4\\) we’d have four rewards plus \\(\\gamma^4 Q(s_{t+4}, a_{t+4})\\).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For \\(n=2\\), write the n-step return \\(G_{t:t+2}\\) in terms of \\(r_{t+1}, r_{t+2}, Q(s_{t+2}, a_{t+2})\\) and \\(\\gamma\\).
2. **Coding:** Write a function that, given a trajectory (s_0, a_0, r_1, s_1, ..., s_T) and n, returns the n-step returns G_{t:t+n} for t=0..T-n. Use a fixed Q-table for bootstrap.
3. **Challenge:** Implement n-step *Q-learning*: use \\(G = r_1 + \\cdots + \\gamma^{n-1} r_n + \\gamma^n \\max_{a} Q(s_n, a)\\). Compare with n-step SARSA on Cliff Walking.
4. **Variant:** Try \\(n = 1, 4, 10\\) on Cliff Walking. Plot learning curves. Which \\(n\\) gives the best performance? Is there a point where larger \\(n\\) hurts?
5. **Debug:** The n-step return below does not handle episode termination correctly — it bootstraps with \\(Q(s_n,a_n)\\) even when \\(s_n\\) is terminal. Fix it.

{{< pyrepl code="def nstep_return(rewards, q_next, gamma, n, done=False):\n    \"\"\"Compute n-step return from rewards list.\"\"\"\n    G = sum(gamma**i * r for i, r in enumerate(rewards[:n]))\n    # BUG: adds bootstrap even when episode is done\n    G += gamma**n * q_next\n    return G\n\n# Fix: only add bootstrap term if not done\n# G += (0 if done else gamma**n * q_next)\nprint(nstep_return([0,0,1], q_next=0.5, gamma=0.9, n=3, done=True))\n# When done=True, should NOT add q_next term" height="220" >}}

6. **Conceptual:** How does n-step bootstrapping unify MC (\\(n=\\infty\\)) and TD(0) (\\(n=1\\))? What is the bias-variance trade-off as \\(n\\) increases?
7. **Recall:** Write the n-step return \\(G_{t:t+n}\\) formula from memory.
