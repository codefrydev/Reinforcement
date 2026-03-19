---
title: "Chapter 2: Multi-Armed Bandits"
description: "10-armed testbed with epsilon-greedy vs greedy."
date: 2026-03-10T00:00:00Z
weight: 2
draft: false
tags: ["multi-armed bandits", "epsilon-greedy", "exploration", "curriculum"]
keywords: ["multi-armed bandits", "epsilon-greedy", "10-armed testbed", "exploration exploitation"]
---

{{< notebook path="volume-01/ch02_bandits.ipynb" title="Open Chapter 2 notebook" >}}

**Learning objectives**

- Implement a multi-armed bandit environment with Gaussian rewards.
- Compare epsilon-greedy and greedy policies in terms of average reward and regret.
- Recognize the exploration–exploitation trade-off in a simple setting.

**Concept and real-world RL**

A **multi-armed bandit** is an RL problem with a single state: the agent repeatedly chooses an "arm" (action) and receives a reward drawn from a distribution associated with that arm. The goal is to maximize cumulative reward. **Exploration** (trying different arms) is needed to discover which arm has the highest mean; **exploitation** (choosing the best arm so far) maximizes immediate reward. In practice, bandits model A/B testing, clinical trials, and recommender systems (which ad or item to show). The **10-armed testbed** is a standard benchmark: 10 arms with different unknown means; the agent learns from experience.

**Exercise:** Implement a 10-armed testbed as in Sutton & Barto, where each arm's reward is drawn from a Gaussian with unit variance and mean \\(\mu_i \sim \mathcal{N}(0,1)\\). Run an epsilon-greedy agent (\\(\epsilon = 0.1\\)) for 1000 steps and plot the average reward over time. Compare with a purely greedy agent.

{{< pyrepl code="import random\nrandom.seed(42)\n\n# 3-armed bandit\ntrue_means = [0.1, 0.5, -0.2]\nQ = [0.0, 0.0, 0.0]  # estimates\nN = [0, 0, 0]         # pull counts\n\n# TODO: run 100 steps of epsilon-greedy (epsilon=0.1)\nepsilon = 0.1\nfor step in range(100):\n    # choose action\n    if random.random() < epsilon:\n        action = random.randrange(3)\n    else:\n        action = Q.index(max(Q))\n    # pull arm\n    reward = random.gauss(true_means[action], 1)\n    # update\n    N[action] += 1\n    Q[action] += (reward - Q[action]) / N[action]\n\nprint('Q estimates:', [round(q,3) for q in Q])\nprint('True means: ', true_means)\nprint('Pull counts:', N)" height="320" >}}

**Professor's hints**

- Sample the 10 means once at the start: `np.random.randn(10)`. Each pull of arm \\(i\\) returns `means[i] + np.random.randn()`.
- Maintain for each arm: number of pulls and running mean (or sum of rewards). Update the running mean incrementally: \\(\\bar{Q}_{n+1} = \\bar{Q}_n + \\frac{1}{n+1}(r - \\bar{Q}_n)\\).
- With probability \\(\epsilon\\) choose a random arm; with probability \\(1-\\epsilon\\) choose \\(\\arg\\max_a Q(a)\\). Break ties arbitrarily (e.g. first max).
- Run many independent runs (e.g. 100 or 200) and plot the **average** reward over time across runs so the curve is smooth. Same for the greedy agent.

**Common pitfalls**

- **Initializing Q too optimistically:** Greedy with Q=0 for all arms will try each arm once, then stick to one. If you initialize Q to large values, greedy explores more early (optimistic initial values). For a fair comparison, use the same initialization for both agents.
- **Using the same random seed for both agents:** Use different runs or the same seed for the *environment* (arm means and rewards) so the comparison is fair; otherwise one agent might get "lucky" arms.
- **Plotting one run only:** One run is very noisy. Average over at least 50–100 runs to see the difference between epsilon-greedy and greedy clearly.

{{< collapse summary="Worked solution (warm-up: optimal arm and greedy choice)" >}}
**Warm-up:** For 3 arms with known means [0.1, 0.5, -0.2], compute the expected reward of the optimal arm. If you pull each arm 10 times and get sample means [0.2, 0.4, -0.1], which arm would greedy choose? Would that be correct?

**Step 1 — Optimal arm:** The expected reward of the optimal arm is the maximum of the true means: \\(\max(0.1, 0.5, -0.2) = 0.5\\) (arm 2).

**Step 2 — Greedy choice:** Greedy chooses \\(\arg\max_a Q(a)\\) using the *sample* means. So greedy picks \\(\arg\max([0.2, 0.4, -0.1]) = \\) arm 2 (index 1). That *is* correct—the best arm in this run’s estimates is also the true best arm.

**Step 3 — When greedy can be wrong:** If the sample means had been [0.6, 0.3, -0.1], greedy would choose arm 1, which has true mean 0.1 (worse than arm 2’s 0.5). So greedy can lock onto a suboptimal arm when early samples are misleading; epsilon-greedy keeps exploring and often finds the true best arm. This is the exploration–exploitation trade-off.
{{< /collapse >}}

The graph below shows the sample means [0.2, 0.4, -0.1] after 10 pulls per arm; greedy picks arm 2 (highest bar). True means were 0.1, 0.5, -0.2.

{{< chart type="bar" title="Sample means Q(a) after 10 pulls (3 arms)" labels="Arm 1, Arm 2, Arm 3" data="0.2, 0.4, -0.1" >}}

**Extra practice**

1. **Warm-up:** For 3 arms with known means [0.1, 0.5, -0.2], compute the expected reward of the optimal arm. If you pull each arm 10 times and get sample means [0.2, 0.4, -0.1], which arm would greedy choose? Would that be correct?
2. **Coding:** Implement epsilon-greedy for a 5-armed bandit with Gaussian rewards (mean 0, variance 1 per arm, with different true means). Run for 1000 steps with ε=0.1 and plot the cumulative regret vs t.
3. **Challenge:** Add a **UCB** (upper-confidence-bound) agent: choose \\(a = \\arg\\max_a \\bigl[ Q(a) + c \\sqrt{\\frac{\\ln t}{N(a)}} \\bigr]\\). Plot UCB alongside epsilon-greedy and greedy for \\(c=2\\).
4. **Variant:** Try \\(\epsilon = 0.01\\) and \\(\epsilon = 0.5\\) on the same 10-armed testbed. Which finds the best arm faster? Which accumulates more regret over 1000 steps?
5. **Debug:** The incremental update below has a bug — it divides by the wrong count. Find and fix it.

{{< pyrepl code="Q = [0.0, 0.0, 0.0]\nN = [0, 0, 0]\n\ndef update(action, reward):\n    N[action] += 1\n    # BUG: should divide by N[action], not total steps\n    total = sum(N)\n    Q[action] += (reward - Q[action]) / total\n\nupdate(1, 1.0)\nupdate(1, 0.0)\nprint('Q[1] after 2 pulls:', Q[1])  # expected 0.5\n# TODO: fix update() so Q[1] == 0.5" height="220" >}}

6. **Conceptual:** Why does a purely greedy agent (\\(\epsilon=0\\)) typically perform worse than \\(\epsilon\\)-greedy in the long run on the 10-armed testbed?
7. **Recall:** State the incremental update formula \\(Q_{n+1} = Q_n + \frac{1}{n+1}(r - Q_n)\\) from memory and explain what each term means.
