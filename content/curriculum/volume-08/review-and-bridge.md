---
title: "Volume 8 Review & Bridge to Volume 9"
description: "Review Volume 8 (Offline RL, Imitation Learning, IRL, RLHF) and preview Volume 9 (Multi-Agent RL — cooperation, competition, game theory)."
date: 2026-03-19T00:00:00Z
draft: false
difficulty: 8
weight: 100
tags: ["review", "bridge", "Volume 8", "Volume 9", "offline RL", "RLHF", "multi-agent", "game theory"]
roadmap_color: "violet"
roadmap_icon: "database"
roadmap_phase_label: "Vol 8 · Review"
---

## Volume 8 Recap Quiz (5 questions)

{{< collapse summary="Q1. What is behavioral cloning (BC) and what is its main failure mode?" >}}
BC treats imitation learning as supervised learning: train a policy π(a|s) to minimize cross-entropy loss against expert actions. Simple and effective when the dataset is large and diverse. **Main failure mode: covariate shift (compounding errors).** A small mistake moves the agent to a state not seen in the expert data; the policy hasn't learned to recover, making further mistakes. After T steps, error can grow as O(T²).
{{< /collapse >}}

{{< collapse summary="Q2. How does DAgger fix the covariate shift problem?" >}}
DAgger (Dataset Aggregation) iteratively: (1) run the current policy to collect states the learner actually visits; (2) query the expert for correct actions at those states; (3) aggregate the new data into the training set; (4) retrain. By training on the **distribution of states the learner visits** (not just expert trajectories), DAgger achieves O(T) error instead of O(T²). It requires an interactive expert.
{{< /collapse >}}

{{< collapse summary="Q3. What is Inverse Reinforcement Learning (IRL), and when is it better than BC?" >}}
IRL infers the reward function R(s,a) that explains expert behaviour, then solves the RL problem with that inferred reward. Better than BC when: (1) you want the agent to generalise to new environments (BC copies actions; IRL recovers goals); (2) the expert data is sparse or suboptimal; (3) you want to transfer the policy to a different body/dynamics. Drawback: IRL is ill-posed (many rewards explain the same behaviour) and computationally expensive.
{{< /collapse >}}

{{< collapse summary="Q4. Describe the RLHF pipeline for LLM alignment (3 stages)." >}}
1. **Supervised Fine-Tuning (SFT)**: fine-tune the base LLM on high-quality human-written demonstrations.
2. **Reward Model Training**: collect human preference comparisons (response A vs B); train a reward model R(prompt, response) to predict human preference scores.
3. **RL Fine-Tuning with PPO**: use PPO to optimise the LLM policy to maximise R, with a KL penalty against the SFT model to prevent reward hacking and mode collapse.
{{< /collapse >}}

{{< collapse summary="Q5. What is Conservative Q-Learning (CQL) and why is it needed for offline RL?" >}}
CQL adds a penalty to the standard Bellman loss that **lowers Q-values for out-of-distribution (OOD) actions** while raising them for in-distribution actions. Formally it adds: α · (E_{a~μ}[Q(s,a)] − E_{a~β}[Q(s,a)]) to the loss. This prevents the Q-function from overestimating value for actions never seen in the dataset. BCQ achieves similar goals by constraining actions to stay close to the behaviour policy.
{{< /collapse >}}

---

## What Changes in Volume 9

| | Volume 8 (Single Agent) | Volume 9 (Multi-Agent) |
|---|---|---|
| **Environment** | One agent, stationary world | Multiple agents, each affecting others |
| **Stationarity** | Environment is fixed | Non-stationary — other agents are learning |
| **Solution concept** | Optimal policy | Nash equilibrium |
| **Credit assignment** | Straightforward | Hard — joint reward, individual actions |
| **Key challenge** | OOD actions, distribution shift | Non-stationarity, communication, emergent behaviour |

**The big insight:** With multiple agents, each agent's optimal policy depends on what the others do — and they're all changing simultaneously. This breaks the Markov assumption for any single agent. Game theory (Nash equilibrium, zero-sum, cooperative) provides the right framework. CTDE (Centralised Training, Decentralised Execution) is the dominant paradigm: train with global info, deploy with local observations.

---

## Bridge Exercise: The Prisoner's Dilemma — Game Theory in Action

{{< pyrepl code="# The Prisoner's Dilemma: simplest multi-agent game\n# Two agents each choose: Cooperate (C) or Defect (D)\n# Payoffs (agent1, agent2):\n#   C,C -> (3,3)   D,C -> (5,0)   C,D -> (0,5)   D,D -> (1,1)\n\npayoff = {\n    ('C','C'): (3,3), ('C','D'): (0,5),\n    ('D','C'): (5,0), ('D','D'): (1,1)\n}\n\nprint('=== Prisoner Dilemma Payoff Matrix ===')\nprint(f'{'':12} Agent2=C    Agent2=D')\nfor a1 in ['C', 'D']:\n    row = f'Agent1={a1}   '\n    for a2 in ['C', 'D']:\n        p = payoff[(a1, a2)]\n        row += f'({p[0]},{p[1]})       '\n    print(row)\n\nprint()\nprint('Nash Equilibrium Analysis:')\nprint('  If Agent2 plays C: Agent1 prefers D (5 > 3)')\nprint('  If Agent2 plays D: Agent1 prefers D (1 > 0)')\nprint('  => D is dominant strategy for both agents')\nprint('  => Nash Equilibrium: (D, D) with payoff (1,1)')\nprint()\nprint('But (C, C) gives (3, 3) -- better for both!')\nprint('Multi-agent RL must learn to escape suboptimal Nash equilibria.')\nprint('Volume 9: CTDE, QMIX, MAPPO tackle this in cooperative settings.')" height="320" >}}

**Next:** [Volume 9: Multi-Agent RL](../volume-09/)
