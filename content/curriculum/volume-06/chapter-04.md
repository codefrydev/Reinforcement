---
title: "Chapter 54: Monte Carlo Tree Search (MCTS)"
description: "MCTS for tic-tac-toe with UCT; play vs random."
date: 2026-03-10T00:00:00Z
weight: 54
draft: false
tags: ["MCTS", "planning", "tree search", "curriculum"]
keywords: ["Monte Carlo Tree Search", "MCTS", "planning", "tree search"]
---

**Learning objectives**

- Implement **MCTS** for a small game (e.g. **tic-tac-toe**): selection (UCT), expansion, simulation (rollout), backpropagation.
- Use **UCT** (Upper Confidence bound for Trees) for node selection: \\(\\frac{Q(s,a)}{N(s,a)} + c \\sqrt{\\frac{\\log N(s)}{N(s,a)}}\\).
- Evaluate **win rate** against a random opponent.

**Concept and real-world RL**

**MCTS** builds a search tree by repeatedly selecting a leaf (UCT), expanding it, doing a random rollout to the end, and backpropagating the result. It does not require a learned value function (though it can use one, as in AlphaZero). In **game AI** (chess, Go, tic-tac-toe), MCTS is used for planning and action selection; it balances exploration (trying undervisited moves) and exploitation (favoring good moves).

**Where you see this in practice:** AlphaGo, AlphaZero, and many game-playing agents use MCTS (with or without a neural network).

**Illustration (MCTS win rate):** As the number of simulations per move increases, MCTS win rate against a random player typically improves. The chart below shows win rate vs simulations per move.

{{< chart type="line" palette="return" title="Win rate vs simulations per move (tic-tac-toe)" labels="100, 500, 1000, 2000, 5000" data="0.7, 0.85, 0.92, 0.97, 0.99" xLabel="Simulations" yLabel="Win rate" >}}

**Exercise:** Implement MCTS for a tic-tac-toe environment. Use UCT for node selection. Let the algorithm play against a random opponent. Evaluate its win rate.

**Professor's hints**

- Selection: from root, follow the child that maximizes UCT until you reach a leaf (unexpanded node). Expand the leaf (add children for all legal actions), run one random rollout from it, backprop the outcome (win/loss/draw) to all nodes on the path.
- UCT: Q = total reward from that action, N = visit count. Use c ≈ 1.4 or tune. Run many simulations (e.g. 1000) per move, then pick the most visited child (or highest Q/N).
- Win rate: play 100 games vs random; MCTS should win almost all (or draw). Report win/draw/loss.

**Common pitfalls**

- **Backpropagating the right value:** In two-player games, negate the result when backpropagating (opponent's win is our loss). For tic-tac-toe, use +1 win, -1 loss, 0 draw; alternate sign per level.
- **Terminal state:** Do not expand a terminal node; backprop immediately with the game result.

{{< collapse summary="Worked solution (warm-up: MCTS)" >}}
**Key idea:** MCTS: (1) Select: traverse from root by UCB until a leaf. (2) Expand: add children of the leaf. (3) Rollout: play to termination (or use a value network). (4) Backprop: update visit counts and values along the path. The value at the root is the expected outcome; we choose the action with highest value. Used in AlphaZero for games.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** What is the role of the exploration term \\(c \\sqrt{\\log N(s)/N(s,a)}\\) in UCT?
2. **Coding:** Implement MCTS for tic-tac-toe. Plot win rate vs number of simulations per move (100, 500, 1000). Does win rate increase?
3. **Challenge:** Implement MCTS for **connect four** (or another game). Compare win rate vs random with tic-tac-toe (harder game, need more simulations).
4. **Variant:** Vary the exploration constant c ∈ {0.1, 0.5, 1.0, 2.0} in UCT. Plot win rate vs c for tic-tac-toe. Is there a range of c that works well, or is it very sensitive?
5. **Debug:** An MCTS implementation always selects the same move at the root even after 1000 simulations. The bug is that the expansion step never creates child nodes — it returns early if a node has been visited before. Describe the correct expansion condition and how to fix it.
6. **Conceptual:** UCT is asymptotically optimal (converges to minimax given enough simulations). Why is this property useful even though we use a finite simulation budget in practice? What determines how many simulations are "enough"?
