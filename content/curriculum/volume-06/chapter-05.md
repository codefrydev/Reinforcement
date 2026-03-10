---
title: "Chapter 55: AlphaZero Architecture"
description: "Mini AlphaZero for tic-tac-toe: NN + MCTS, self-play."
date: 2026-03-10T00:00:00Z
weight: 55
draft: false
---

**Learning objectives**

- Implement a **simplified AlphaZero** for tic-tac-toe: a **neural network** that outputs **policy** (move probabilities) and **value** (expected outcome).
- Use the network inside **MCTS**: use policy for prior in expansion, value for leaf evaluation (replacing random rollout).
- **Train via self-play**: generate games, train the network on (state, policy target, value target), repeat.

**Concept and real-world RL**

**AlphaZero** combines MCTS with a neural network: the network provides a **prior** over moves and a **value** for leaf states, so MCTS does not need random rollouts. Training is **self-play**: the current network plays against itself; the MCTS policy and game outcome become targets for the network. In **game AI** (chess, Go, shogi), AlphaZero achieves superhuman play. The same idea (planning with a learned model/value) appears in **robot planning** and **dialogue**.

**Where you see this in practice:** AlphaZero, MuZero; game engines and research.

**Exercise:** (Simplified) Implement a mini AlphaZero for tic-tac-toe: combine a neural network that outputs policy probabilities and a value, with MCTS. Train via self-play. Visualize the improvement over training.

**Professor's hints**

- Network: input board (e.g. 9 values or 3×3), output policy (9 dims for 9 cells, mask illegal), and value (1 dim, in [-1,1]). Loss = cross-entropy(policy, MCTS visit counts) + MSE(value, game outcome).
- MCTS: at expansion, use network policy as prior (P(s,a)); at leaf, use network value instead of random rollout. Backprop as usual.
- Self-play: play N games with current network (MCTS for both sides), store (state, π_MCTS, z) for each position; train on a batch; repeat. Plot win rate vs random (or vs previous checkpoint) over training.

**Common pitfalls**

- **Visit counts as policy target:** Normalize MCTS visit counts at the root to get a probability distribution; that is the policy target for the root state.
- **Value target:** Use the outcome from the perspective of the player to move at that state (win=1, loss=-1, draw=0).

**Extra practice**

1. **Warm-up:** Why does AlphaZero use MCTS policy as the training target instead of the raw network policy?
2. **Coding:** Implement mini AlphaZero for tic-tac-toe. Every 100 self-play games, evaluate vs random (20 games). Plot win rate over training.
3. **Challenge:** Add a **replay buffer** of past games (e.g. last 10k positions) and sample from it for training. Does it improve stability?
