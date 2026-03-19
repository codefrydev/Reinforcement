---
title: "Chapter 89: Self-Play and League Training"
description: "Self-play in Tic-Tac-Toe; track ELO."
date: 2026-03-10T00:00:00Z
weight: 89
draft: false
tags: ["self-play", "league training", "Tic-Tac-Toe", "ELO", "curriculum"]
keywords: ["self-play", "league training", "Tic-Tac-Toe", "ELO", "MARL"]
---

**Learning objectives**

- **Implement** **self-play** in a simple game (e.g. Tic-Tac-Toe): two copies of the same agent (or two agents with shared or separate parameters) play against each other; update the policy from the outcomes.
- **Update** both agents (or the single policy) so that they improve against the current opponent (themselves).
- **Track** an **ELO rating** (or win rate vs a fixed baseline) as training progresses to measure improvement.
- **Explain** why self-play can lead to stronger policies (the opponent is always at the current level) and potential pitfalls (cycling, forgetting past strategies).
- **Relate** self-play to **game AI** (AlphaGo, Dota) and **dialogue** (negotiation, debate).

**Concept and real-world RL**

**Self-play** means training by having the agent (or multiple copies) play against itself. Both sides are updated from the game outcomes (e.g. policy gradient for the winner, or both get reward based on win/loss). This creates a **curriculum**: the opponent is always at the current level, so the agent is constantly challenged. **ELO** (or similar) tracks strength over time by comparing win rate against a fixed set of checkpoints or a population. In **game AI** (AlphaGo, Dota, StarCraft), self-play and league training (maintaining a population of opponents) have been key to superhuman performance. In **dialogue**, self-play can train negotiation or debate agents.

**Where you see this in practice:** AlphaGo self-play; OpenAI Five; league training in multi-agent games.

**Illustration (ELO over self-play):** As training progresses, the agent's ELO (or win rate vs random) typically increases. The chart below shows ELO (or win rate × 100) over self-play games.

{{< chart type="line" palette="return" title="Win rate vs random (self-play training)" labels="0, 1k, 2k, 3k, 4k, 5k" data="50, 65, 78, 88, 94, 98" xLabel="Game" yLabel="Win rate (%)" >}}

**Exercise:** Implement self-play in a simple game like Tic-Tac-Toe. Let two copies of an agent play against each other, and update both. Track the ELO rating as training progresses.

**Professor's hints**

- **Tic-Tac-Toe:** 3×3 grid; two players alternate; state = board (e.g. 9 values: 0 empty, 1 P1, 2 P2). Actions = 9 cells. Reward: +1 win, -1 loss, 0 draw (or 0 for draw). Terminal when win or draw.
- **Self-play:** One policy π(a|s) for both players (symmetric). When it's P1's turn, use π(·|s); when P2's turn, use π(·|s') where s' might be from P2's view (e.g. swap 1 and 2 in the board so the policy sees "I am always player 1"). After a game, compute return for each player (e.g. +1 for winner, -1 for loser) and update π with policy gradient (REINFORCE or PPO).
- **ELO:** Maintain a rating R. After a game, update R using the result and the opponent's rating (if you have a fixed opponent or a pool). Or track win rate vs a **random** agent or an **older checkpoint** every N games. Plot R or win rate vs games.
- **Update both:** If using one shared π, one gradient step uses both players' trajectories (with opposite signs for reward). If using two separate policies, update both with their respective rewards.

**Common pitfalls**

- **State representation:** For symmetry, the same player should see the same input (e.g. "my pieces = 1, opponent = -1" and swap for the other player). Otherwise the policy may learn asymmetric play.
- **Draw handling:** Tic-Tac-Toe has many draws once both play well; reward draw as 0 and ensure the policy gets a gradient (e.g. small reward for draw so it learns to avoid loss).
- **ELO implementation:** Standard ELO: expected score = 1/(1+10^((R_opp-R_self)/400)); update R_self += K * (actual_score - expected_score). For a single agent, you can track R vs a fixed random agent.

{{< collapse summary="Worked solution (warm-up: ELO)" >}}
**Key idea:** ELO rates agents by pairwise comparison: expected score of A vs B is \\(1/(1+10^{(R_B - R_A)/400})\\). After a game, update \\(R_A += K (\\text{actual} - \\text{expected})\\). So we get a single number per agent that reflects strength; we can rank many agents without a round-robin. Used in games and to track progress in self-play (e.g. AlphaStar). For MARL we can report ELO vs a baseline pool.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why might an agent trained only against a random opponent fail against a copy of itself? Why does self-play help?
2. **Coding:** Implement Tic-Tac-Toe and self-play with a single shared policy (REINFORCE or PPO). Train for 5k games. Every 500 games, evaluate: win rate vs random, vs previous checkpoint. Plot win rate vs random and vs self (previous) over training.
3. **Challenge:** Implement **league training**: keep a pool of K past policies (checkpoints). Each training game, with probability p play against the current policy and with probability 1-p play against a random choice from the pool. Does this improve robustness and ELO compared to pure self-play?
4. **Variant:** Change the checkpoint save frequency from every 500 games to every 50 games. Does a denser pool of opponents help or hurt training stability? What is the trade-off between policy pool diversity and storage/sampling cost?
5. **Debug:** Self-play training collapses: the agent learns to exploit a specific weakness of its previous self and cycles back and forth between two strategies. ELO stays flat after 3k games. The fix is to periodically reset the opponent to a random earlier checkpoint. Explain why cyclic strategies arise in pure self-play and how a diverse opponent pool breaks the cycle.
6. **Conceptual:** Self-play is a form of curriculum learning where the difficulty of the opponent increases with the learner's skill. Explain why this automatic curriculum is beneficial. What can go wrong if the opponent updates too fast (overfitting to a specific strategy) or too slow (agent never sees a challenging opponent)?
