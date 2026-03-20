---
title: "Chapter 12: Temporal Difference (TD) Learning"
description: "TD(0) prediction for blackjack; compare with Monte Carlo."
date: 2026-03-10T00:00:00Z
weight: 12
draft: false
difficulty: 6
tags: ["temporal difference", "TD", "blackjack", "prediction", "curriculum"]
keywords: ["temporal difference learning", "TD(0)", "blackjack", "bootstrap"]
roadmap_color: "teal"
roadmap_icon: "book"
roadmap_phase_label: "Vol 2 · Ch 2"
---

**Learning objectives**

- Implement TD(0) prediction: update \\(V(s)\\) using the TD target \\(r + \\gamma V(s')\\) immediately after each transition.
- Compare TD(0) with Monte Carlo in terms of convergence speed and sample efficiency.
- Understand bootstrapping: TD uses current estimates instead of waiting for episode end.

**Concept and real-world RL**

**Temporal Difference (TD) learning** updates value estimates using the **TD target** \\(r + \\gamma V(s')\\): \\(V(s) \\leftarrow V(s) + \\alpha [r + \\gamma V(s') - V(s)]\\). Unlike Monte Carlo, TD does not need to wait for the episode to end; it **bootstraps** on the current estimate of \\(V(s')\\). TD(0) often converges faster per sample and works in continuing tasks. In practice, TD is the basis for SARSA, Q-learning, and many deep RL algorithms (e.g. DQN uses a TD-like target). Blackjack lets you compare TD(0) and MC on the same policy and state space.

**Monte Carlo vs TD bootstrapping:**

{{< mermaid >}}
flowchart TD
    subgraph mc [Monte Carlo]
        direction LR
        s0_mc["s_0"] --> s1_mc["s_1"] --> dots_mc["..."] --> sT_mc["s_T (terminal)"]
        sT_mc -->|"G = r_0 + γr_1 + ... + r_T"| update_mc["Update V(s_0)"]
    end
    subgraph td [TD Learning]
        direction LR
        s0_td["s_t"] --> s1_td["s_{t+1}"]
        s1_td -->|"δ = r_t + γV(s_{t+1}) - V(s_t)"| update_td["Update V(s_t)"]
    end
{{< /mermaid >}}

**Illustration (TD convergence):** TD(0) updates \\(V(s)\\) after every transition. The estimate for a given state often stabilizes faster than in MC because TD bootstraps. The chart below shows a typical trend: \\(V(s)\\) for one state over the first 20 episodes.

{{< chart type="line" palette="return" title="V(s) over episodes (TD(0), α=0.01)" labels="0, 5, 10, 15, 20" data="0, 0.3, 0.55, 0.72, 0.85" xLabel="Episode" yLabel="V(s)" >}}

**Exercise:** Implement TD(0) prediction for the same blackjack policy. Compare the convergence speed and final value estimates with Monte Carlo. Use a step size \\(\alpha=0.01\\) and run for 10,000 episodes.

**Professor's hints**

- Initialize \\(V(s)=0\\) for all states (or a default). For each transition \\((s, a, r, s')\\), update \\(V(s) \\leftarrow V(s) + \\alpha [r + \\gamma V(s') - V(s)]\\). Use the *updated* \\(V\\) for the next step (online update).
- You need to run episodes and, at each step, have \\(s\\), \\(r\\), \\(s'\\). The policy is fixed (stick on 20/21, else hit). Same blackjack env as Chapter 11.
- Comparison: run MC for 10k episodes and TD for 10k episodes. Plot \\(V(s)\\) for a few states over episodes (e.g. log scale or every N episodes). TD often stabilizes faster; final estimates may be slightly different because TD is biased (bootstrapping) while MC is unbiased.

**Common pitfalls**

- **Step size:** \\(\alpha=0.01\\) is small; values change slowly. If you use a large \\(\alpha\\), TD can be noisy or unstable. For tabular blackjack, 0.01–0.1 is typical.
- **Comparing fairly:** Use the same number of episodes (10k) for both. MC needs full episodes; TD updates every step. So TD makes many more updates per episode—keep that in mind when comparing "convergence speed."
- **Initial V(s'):** For terminal \\(s'\\), \\(V(s')=0\\) by definition. Do not update \\(V\\) for terminal states; use 0 when computing the target.

{{< collapse summary="Worked solution (warm-up: TD(0) update in one line)" >}}
**Warm-up:** Write the TD(0) update in one line: given \\(s, r, s', V, \\alpha, \\gamma\\), what is the new \\(V(s)\\)? **Answer:** \\(V(s) \\leftarrow V(s) + \\alpha [r + \\gamma V(s') - V(s)]\\). In Python: `V[s] = V[s] + alpha * (r + gamma * V[s'] - V[s])`. The TD error is \\(\\delta = r + \\gamma V(s') - V(s)\\); we move \\(V(s)\\) in the direction of the error. Unlike MC, we don’t need the full return—we bootstrap on \\(V(s')\\).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Write the TD(0) update in one line of Python (pseudo-code): given \\(s, r, s', V, \\alpha, \\gamma\\), what is the new \\(V(s)\\)?
2. **Coding:** Implement TD(0) prediction for a fixed policy on a small tabular MDP (e.g. 4 states). Run for 1000 episodes and plot V(s) for one state over time.
3. **Challenge:** Run TD(0) with \\(\alpha \\in \\{0.001, 0.01, 0.1\\}\\) for 10k episodes. Plot the learning curve (e.g. mean absolute error from a reference \\(V\\) if you have one, or \\(V\\) for one state over time). Which \\(\\alpha\\) converges fastest? Which is most stable?
4. **Variant:** Run TD(0) with \\(\gamma=0.5\\) and \\(\gamma=0.99\\) on the same tabular MDP. How do the value estimates for the same state differ? Which \\(\gamma\\) makes values converge faster?
5. **Debug:** The code below forgets to use \\(V(s') = 0\\) for terminal transitions (done=True), causing values near terminal states to be incorrect. Fix it.

{{< pyrepl code="V = {'A': 0.0, 'B': 0.0, 'terminal': 0.0}\nalpha, gamma = 0.1, 0.9\n\ndef td_update(s, r, s_next, done):\n    # BUG: does not handle terminal state\n    td_target = r + gamma * V[s_next]  # wrong when done=True\n    V[s] = V[s] + alpha * (td_target - V[s])\n\n# Fix: when done=True, target should be just r\ntd_update('A', 1, 'terminal', done=True)\nprint('V[A]:', V['A'])  # should be ~0.1 (alpha * (1 - 0))\nprint('Fix: use target = r when done=True')" height="220" >}}

6. **Conceptual:** Why does TD(0) have lower variance than MC but higher bias? When would you prefer one over the other?
7. **Recall:** Write the TD(0) update rule \\(V(s) \\leftarrow \\ldots\\) from memory.
