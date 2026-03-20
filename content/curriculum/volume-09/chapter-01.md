---
title: "Chapter 81: Multi-Agent Fundamentals"
description: "Model Rock-Paper-Scissors as Dec-POMDP."
date: 2026-03-10T00:00:00Z
weight: 81
draft: false
difficulty: 8
tags: ["multi-agent", "MARL", "Dec-POMDP", "game theory", "curriculum"]
keywords: ["multi-agent RL", "Dec-POMDP", "Rock-Paper-Scissors", "game theory"]
roadmap_color: "blue"
roadmap_icon: "network"
roadmap_phase_label: "Vol 9 · Ch 1"
---

**Learning objectives**

- **Model** a two-player zero-sum game (e.g. Rock-Paper-Scissors) as a **Dec-POMDP** (Decentralized Partially Observable MDP) or equivalent multi-agent framework.
- **Define** states, observations, actions, and rewards for each agent in the game.
- **Explain** the difference between centralized (one controller sees everything) and decentralized (each agent has its own observation and policy) formulations.
- **Identify** how the same game can be viewed as a normal-form game (payoff matrix) and as a sequential Dec-POMDP (if we add structure).
- **Relate** multi-agent modeling to **game AI** (opponents, teammates) and **trading** (multiple market participants).

**Concept and real-world RL**

**Multi-agent RL** studies settings where multiple **agents** act in a shared environment; each agent has its own observations and actions and typically its own policy. A **Dec-POMDP** extends the MDP with multiple agents: each agent has a local observation (possibly partial) and chooses an action; the state transitions and rewards can depend on all agents' actions. A **zero-sum game** (e.g. Rock-Paper-Scissors) has two agents whose rewards sum to zero. Modeling it as a Dec-POMDP makes the state (e.g. "no prior moves" or "last move of each"), observations (e.g. own action, or nothing until reveal), and rewards (e.g. +1 win, -1 lose, 0 draw) explicit. In **game AI** and **trading**, multiple agents (players, bots, traders) interact; Dec-POMDPs and game-theoretic models are the foundation.

**Where you see this in practice:** Game theory and multi-agent systems; Dec-POMDPs in robotics and games; Nash equilibrium and learning in games.

**Illustration (zero-sum payoffs):** In Rock-Paper-Scissors, one player's gain is the other's loss. The chart below shows payoff to row player for each outcome (R-R draw, R-P loss, R-S win, etc.).

{{< chart type="bar" title="Payoff to row player (RPS outcomes)" labels="R-R, R-P, R-S, P-R, P-P, P-S" data="0, -1, 1, 1, 0, -1" >}}

**Exercise:** Model a two-player zero-sum game (e.g., Rock-Paper-Scissors) as a Dec-POMDP. Define states, observations, actions, and rewards.

**Professor's hints**

- **States:** For RPS, the state can be "before move" (both choose simultaneously) or "after move" (outcome determined). In a one-shot game, state = (none) or (action_1, action_2) for outcome. For a repeated game, state could include history of plays.
- **Observations:** Each agent may see only its own action (or nothing until the end). So agent i's observation could be o_i = (nothing) until reveal, then (a_1, a_2, r_i). Or in a sequential version, each agent observes the other's past action.
- **Actions:** Each agent: {Rock, Paper, Scissors}. Joint action (a_1, a_2) determines reward: r_1 = -r_2; +1 win, -1 lose, 0 draw.
- **Rewards:** Define r_1(s, a_1, a_2) and r_2 = -r_1. Write a small table or function.
- You can present this as a short write-up with a state space, observation spaces, action spaces, and reward function; no code required, but code for the env is good practice.

**Common pitfalls**

- **Confusing state and observation:** In a Dec-POMDP, the state can be global (e.g. both actions); each agent's observation may be a function of the state (e.g. only own action). Be explicit.
- **Simultaneous vs sequential:** RPS is simultaneous; the "state" before actions does not include the other's action. In sequential games, state would include whose turn and past actions.
- **Zero-sum:** Ensure r_1 + r_2 = 0 for all outcomes.

{{< collapse summary="Worked solution (warm-up: two-agent zero-sum)" >}}
**Key idea:** In two-agent zero-sum games, one agent’s gain is the other’s loss (\\(r_1 + r_2 = 0\\)). The Nash equilibrium is a pair of policies where neither can improve by deviating. We can train with self-play: each agent (or a single policy that sees "which player am I") tries to maximize its own return; the equilibrium emerges when both play best response. Used in games (chess, Go) and adversarial settings.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In Rock-Paper-Scissors, what is a minimal state space? What can each agent observe at decision time?
2. **Coding:** Implement a Rock-Paper-Scissors environment: two agents, actions in {0,1,2}, rewards from the payoff matrix. Step returns (s, o_1, o_2, a_1, a_2, r_1, r_2, done). Run 100 random plays and verify reward sums to zero.
3. **Challenge:** Model a **repeated** RPS game with a fixed number of rounds. State = history of (a_1, a_2). Each agent observes only the history (or only own actions and outcomes). Define the horizon and total return (e.g. sum of rewards over rounds).
4. **Variant:** Change from zero-sum (RPS) to a cooperative game: both agents receive +1 only if they choose the same action. How does the reward structure change the dynamics and equilibrium compared to the zero-sum case?
5. **Debug:** A multi-agent environment returns `rewards` as a single scalar shared between agents, but one agent gets + and the other gets - of the same value. The reward assignment is missing per-agent indexing. Fix the reward structure and explain why sharing one scalar breaks independent learning algorithms.
6. **Conceptual:** In single-agent RL, the environment is stationary. In multi-agent RL, each agent's environment is non-stationary because other agents are learning simultaneously. Explain why this violates the Markov assumption for each individual agent and what convergence guarantees (if any) survive.
