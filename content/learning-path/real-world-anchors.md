---
title: "Real-World Scenarios in This Curriculum"
description: "Anchor scenarios used throughout the learning path and curriculum to ground RL concepts in practice."
date: 2026-03-10T00:00:00Z
draft: false
difficulty: 0
tags: ["real-world", "anchor scenarios", "robot navigation", "game AI", "recommendation", "learning path"]
keywords: ["real-world RL scenarios", "anchor scenarios", "robot navigation", "game AI", "recommendation", "trading", "healthcare", "dialogue"]
roadmap_icon: "globe"
roadmap_color: "blue"
roadmap_phase_label: "Real-World Anchors"
---

Throughout this curriculum we refer to **anchor scenarios**—concrete real-world settings where reinforcement learning is used. These help you see how each concept (MDPs, value functions, policy gradients, etc.) appears in practice. When you see a concept, ask: "How does this show up in robot navigation? In game AI? In recommendation?"

---

## Anchor scenarios

| Scenario | What it is | State / action / reward (typical) | Where it appears in the curriculum |
|----------|------------|-----------------------------------|------------------------------------|
| **Robot navigation** | A robot or agent moves in a physical or simulated space to reach a goal. | State: position, velocity, sensor readings. Action: move, turn. Reward: +1 at goal, small cost per step or collision. | Vol 1–2 (gridworld, MDPs, value iteration); Vol 3–5 (DQN, policy gradients for continuous control). |
| **Game AI** | An agent plays a game (board game, video game, card game) with rules and opponents. | State: board position or game screen; action: move, play card; reward: win/loss or score. | Vol 1 (return, discounting); Vol 2 (blackjack MC, Q-learning); Vol 3–4 (DQN, policy gradients); Vol 7 (exploration). |
| **Recommendation** | A system suggests items (videos, products, articles) to users; goal is long-term engagement or satisfaction. | State: user history, context; action: which item to show; reward: click, watch time, or purchase. | Vol 1–2 (bandits, MDP for sequential decisions); Vol 8 (offline RL from logs); Vol 10 (real-world RL). |
| **Trading / finance** | An agent makes buy/sell/hold decisions in markets with uncertain outcomes. | State: prices, portfolio, indicators; action: trade or hold; reward: profit, risk-adjusted return. | Vol 1 (delayed reward, discounting); Vol 6 (model-based); Vol 10 (safety, real-world). |
| **Healthcare / dosing** | Decisions about treatment, dosage, or interventions over time with safety constraints. | State: patient history, vitals; action: dose or intervention; reward: outcome minus harm. | Vol 1 (MDP, reward design); Vol 8 (offline RL from historical data); Vol 10 (safety, constraints). |
| **Dialogue / assistants** | An agent (chatbot, voice assistant) chooses responses to maximize user satisfaction or task completion. | State: conversation history, user intent; action: response or API call; reward: user feedback, task success. | Vol 4–5 (policy gradients, PPO); Vol 10 (RLHF, LLMs). |

---

## How we use these in chapters

- **Concept and real-world RL:** Each chapter ties the concept to at least one anchor (e.g. "In robot navigation, the state is (position, velocity); in recommendation, the state can be user history.").
- **Where you see this in practice:** Some chapters add a short list, e.g. "Used in: AlphaGo (MDP), industrial control (value iteration)."
- **Exercises:** When an exercise is generic (e.g. "implement Q-learning"), you can re-use the same code on a gridworld (robot-like) or a simple game (game AI). The math is the same; the scenario gives context.

---

## Quick reference by volume

- **Vol 1 (Mathematical foundations):** Gridworld and bandits → robot navigation, game AI, recommendation (multi-armed bandits).
- **Vol 2 (Tabular methods):** Blackjack, gridworld → game AI, robot navigation.
- **Vol 3 (Value approximation, DQN):** CartPole, Atari-style → robot control, game AI.
- **Vol 4–5 (Policy gradients, PPO):** CartPole, MuJoCo → robot control, dialogue, game AI.
- **Vol 6 (Model-based):** Planning, Dreamer → robot navigation, trading.
- **Vol 7 (Exploration):** Bandits, curiosity → recommendation, game AI.
- **Vol 8 (Offline / imitation):** Batch data → recommendation, healthcare.
- **Vol 9 (Multi-agent):** Multiple agents → game AI, trading, dialogue.
- **Vol 10 (Real-world, safety, LLMs):** All anchors; safety for healthcare and finance; RLHF for dialogue.

Use these anchors to make the curriculum concrete. When in doubt, map the current chapter to one of the six and ask: "What is the state? The action? The reward?"
