---
title: "Chapter 55: AlphaZero Architecture"
description: "Mini AlphaZero for tic-tac-toe: NN + MCTS, self-play."
date: 2026-03-10T00:00:00Z
weight: 55
draft: false
difficulty: 8
tags: ["AlphaZero", "MCTS", "tic-tac-toe", "self-play", "curriculum"]
keywords: ["AlphaZero", "MCTS", "neural network", "self-play", "tic-tac-toe"]
roadmap_color: "rose"
roadmap_icon: "brain"
roadmap_phase_label: "Vol 6 · Ch 5"
---

**Learning objectives**

- Implement a **simplified AlphaZero** for tic-tac-toe: a **neural network** that outputs **policy** (move probabilities) and **value** (expected outcome).
- Use the network inside **MCTS**: use policy for prior in expansion, value for leaf evaluation (replacing random rollout).
- **Train via self-play**: generate games, train the network on (state, policy target, value target), repeat.

**Concept and real-world RL**

**AlphaZero** combines MCTS with a neural network: the network provides a **prior** over moves and a **value** for leaf states, so MCTS does not need random rollouts. Training is **self-play**: the current network plays against itself; the MCTS policy and game outcome become targets for the network. In **game AI** (chess, Go, shogi), AlphaZero achieves superhuman play. The same idea (planning with a learned model/value) appears in **robot planning** and **dialogue**.

**Where you see this in practice:** AlphaZero, MuZero; game engines and research.

**Illustration (self-play improvement):** AlphaZero-style training improves win rate over iterations as the policy and value network get better. The chart below shows win rate vs self-play iteration (vs previous checkpoint).

{{< chart type="line" palette="return" title="Win rate vs self-play iteration" labels="0, 50, 100, 150, 200" data="0.5, 0.65, 0.78, 0.88, 0.94" xLabel="Iteration" yLabel="Win rate" >}}

**Exercise:** (Simplified) Implement a mini AlphaZero for tic-tac-toe: combine a neural network that outputs policy probabilities and a value, with MCTS. Train via self-play. Visualize the improvement over training.

**Professor's hints**

- Network: input board (e.g. 9 values or 3×3), output policy (9 dims for 9 cells, mask illegal), and value (1 dim, in [-1,1]). Loss = cross-entropy(policy, MCTS visit counts) + MSE(value, game outcome).
- MCTS: at expansion, use network policy as prior (P(s,a)); at leaf, use network value instead of random rollout. Backprop as usual.
- Self-play: play N games with current network (MCTS for both sides), store (state, π_MCTS, z) for each position; train on a batch; repeat. Plot win rate vs random (or vs previous checkpoint) over training.

**Common pitfalls**

- **Visit counts as policy target:** Normalize MCTS visit counts at the root to get a probability distribution; that is the policy target for the root state.
- **Value target:** Use the outcome from the perspective of the player to move at that state (win=1, loss=-1, draw=0).

{{< collapse summary="Worked solution (warm-up: AlphaZero-style)" >}}
**Key idea:** Train a network that takes state and outputs (policy \\(p\\), value \\(v\\)). Use MCTS with the policy as prior and the value at leaves; the MCTS visit distribution is the target policy. Loss = cross-entropy(p, MCTS_policy) + MSE(v, game outcome). Self-play generates data; the network improves and MCTS improves with it. This is the core of AlphaZero.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why does AlphaZero use MCTS policy as the training target instead of the raw network policy?
2. **Coding:** Implement mini AlphaZero for tic-tac-toe. Every 100 self-play games, evaluate vs random (20 games). Plot win rate over training.
3. **Challenge:** Add a **replay buffer** of past games (e.g. last 10k positions) and sample from it for training. Does it improve stability?
4. **Variant:** Change the number of MCTS simulations per move from 50 to 200 and to 10. How does win rate and training speed trade off? At what point does fewer simulations hurt policy quality?
5. **Debug:** Mini AlphaZero trains but win rate plateaus at 50% (no better than random). The network outputs the same prior probability for all moves. Identify two possible causes: one related to network initialization and one related to the loss function.
6. **Conceptual:** AlphaZero improves its own training targets over time via self-play. Why could this feedback loop become unstable (the "forgetting" problem), and how does the replay buffer mitigate it?
