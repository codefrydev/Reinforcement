---
title: "Chapter 82: Game Theory Basics for RL"
description: "Nash equilibrium of 2×2 matrix; independent learning outcome."
date: 2026-03-10T00:00:00Z
weight: 82
draft: false
---

**Learning objectives**

- **Compute** the **Nash equilibrium** of a simple 2×2 game (e.g. Prisoner's Dilemma) from the payoff matrix.
- **Explain** why **independent learning** (each agent learns its best response without knowing the other's policy) might converge to an outcome that is **not** a Nash equilibrium, or might not converge at all.
- **Compare** Nash equilibrium payoffs with the payoffs that result from independent Q-learning or gradient-based learning in the same game.
- **Identify** the difference between cooperative, competitive, and mixed settings in terms of payoff structure.
- **Relate** game theory to **game AI** (opponent modeling) and **trading** (market equilibrium).

**Concept and real-world RL**

**Game theory** provides solution concepts (e.g. **Nash equilibrium**) for multi-agent settings: at Nash, no agent can improve its payoff by unilaterally changing its strategy. A **2×2 matrix game** (two agents, two actions each) has a simple payoff matrix; Nash equilibria can be pure (one action per agent) or mixed (randomize over actions). In **independent learning**, each agent updates its policy based on its own experience without explicitly modeling the other; this can lead to **non-stationarity** (the other agent's policy changes) and convergence to non-Nash outcomes (e.g. mutual defection in Prisoner's Dilemma even when both could do better). In **game AI** and **trading**, understanding Nash and learning dynamics helps design and analyze multi-agent systems.

**Where you see this in practice:** Nash equilibrium in games and economics; independent learning and self-play; convergence issues in MARL.

**Exercise:** Compute the Nash equilibrium of a simple 2×2 payoff matrix (e.g., Prisoner's Dilemma). Explain why independent learning might converge to a different outcome.

**Professor's hints**

- **Prisoner's Dilemma:** Rows = agent 1 (Cooperate, Defect), columns = agent 2. Typical payoffs: (C,C)=(−1,−1), (C,D)=(−3,0), (D,C)=(0,−3), (D,D)=(−2,−2). Nash equilibrium (in pure strategies) is (D,D): each player's best response to the other's D is D. But (C,C) is better for both—the "dilemma."
- **Computing Nash:** For pure strategies, check each joint action: is it a best response for both? For mixed strategies (2×2), set up indifference equations: agent 1's mix (p, 1-p) such that agent 2 is indifferent between its actions; solve for p and q.
- **Independent learning:** If both agents run Q-learning or gradient ascent, they may both learn to Defect (converge to (D,D)) because each is best-responding to the current policy of the other. They do not coordinate to (C,C). Explain this in 2–3 sentences.
- Optionally run a simple simulation: two Q-learning agents in the matrix game; log their policies over time. Do they converge to Nash?

**Common pitfalls**

- **Payoff order:** In a payoff matrix, (row, col) often means (agent1, agent2). Check the convention (who is row, who is column) and stick to it.
- **Multiple equilibria:** Some games have more than one Nash equilibrium; mention which one(s) you found.
- **Independent learning ≠ Nash:** Independent learners do not necessarily converge to Nash; they may cycle or converge to a different outcome. The exercise asks you to explain this possibility.

**Extra practice**

1. **Warm-up:** In the Prisoner's Dilemma, what is agent 1's best response if agent 2 plays Cooperate? If agent 2 plays Defect? So what is the Nash equilibrium?
2. **Coding:** Implement two independent Q-learning agents in a 2×2 matrix game (Prisoner's Dilemma). Run for 5000 episodes; each episode = one simultaneous action choice. Plot each agent's probability of Defect over time. Do they converge to (D,D)?
3. **Challenge:** Find the **mixed-strategy** Nash equilibrium of a 2×2 game (e.g. Matching Pennies: (1,-1), (-1,1) for same/different). Compute it by hand and verify with a small script that checks best responses.
