---
title: "Stock Trading Project with Reinforcement Learning"
description: "Apply Q-learning and function approximation to a simplified stock trading environment—data, Q-model, design, and code."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["project", "stock trading", "Q-learning", "reinforcement learning"]
keywords: ["stock trading", "Q-learning", "RL project", "trading"]
---

**Beginners, halt!** If you skipped ahead: this project assumes you have completed the core curriculum through **temporal difference learning** and **approximation methods** (e.g. [Volume 2](../curriculum/volume-02/) and [Volume 3](../curriculum/volume-03/) or equivalent). You should understand Q-learning, state and action spaces, and at least linear function approximation. If you have not done that yet, start with the [Learning path](../learning-path/) and [Course outline](../course-outline/).

---

## Stock Trading Project Section Introduction

This project walks you through building a **simplified RL-based stock trading agent**: you define an environment (state = market/position info, actions = buy/sell/hold), a reward (e.g. profit or risk-adjusted return), and train an agent using Q-learning with function approximation. The goal is to understand how to go from theory (Q-learning, FA) to a concrete design and code.

---

## Data and Environment

**Data:** Use historical price data (e.g. daily OHLCV: open, high, low, close, volume). You can use a single stock or a small universe. Split into train/validation (e.g. by time) so you do not leak future data into training.

**Environment:** The **state** can include: recent returns, position (e.g. long/flat/short or size), cash, maybe simple technical features (moving average, RSI, etc.). The **action** is often discrete: e.g. buy, sell, hold (3 actions) or a small set of position sizes. **Reward:** e.g. change in portfolio value from step to step, or Sharpe-like reward (return minus penalty for volatility). **Transition:** Given state and action, the next state is determined by the next row of data (price move) and your action (position update). So the environment is essentially a **historical replay** of the market with your actions affecting position and PnL.

Design the state so it is (approximately) Markov: include enough history or features that the next state and reward depend only on the current state and action, not the full past.

---

## How to Model Q for Q-Learning

- **State space:** Continuous or high-dimensional (e.g. vector of returns and features). So we **approximate** \\(Q(s,a)\\) with a function: linear \\(Q(s,a) = w^T \phi(s,a)\\) or a small neural network that takes \\(s\\) and outputs one value per action.
- **Action space:** Discrete (e.g. buy/sell/hold). So we have \\(Q(s, a)\\) for each \\(a\\).
- **Updates:** Use the Q-learning update: \\(Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]\\). With function approximation: \\(w \leftarrow w + \alpha \delta \nabla_w Q(s,a;w)\\) where \\(\delta = r + \gamma \max_{a'} Q(s',a';w) - Q(s,a;w)\\) (semi-gradient: do not backprop through the target).
- **Exploration:** Epsilon-greedy or similar during training.

---

## Design of the Program

1. **Data loading:** Load prices, compute returns and any features; split train/validation.
2. **Environment class:** `reset()` returns initial state (e.g. first window of data, zero position). `step(action)` updates position, advances to next time step, returns next state, reward (e.g. PnL change), done (end of data or max steps).
3. **Agent:** Maintains \\(Q\\) (weights or network). Selects action (epsilon-greedy), takes step, computes TD target and updates \\(Q\\).
4. **Training loop:** For each episode (e.g. one pass over train data or a random segment), run the agent until done; log returns and losses.
5. **Evaluation:** On validation data, run with greedy policy (no exploration), report total return, Sharpe, or other metrics.

---

## Code

**Code pt 1 — Data and features:** Load OHLCV, compute simple features (returns, maybe moving averages). Build arrays or DataFrames for train/validation.

**Code pt 2 — Environment:** Implement `TradingEnv`: state = (feature window, position, cash), action = buy/sell/hold (or discrete sizes). Step: apply action to position, advance time, get next row of data, compute reward (e.g. PnL step), return next state and done.

**Code pt 3 — Q-function and agent:** Implement linear \\(Q(s,a) = w^T \phi(s,a)\\) or a small MLP. Agent: `act(s)` returns epsilon-greedy action; `update(s, a, r, s')` does one Q-learning update (semi-gradient).

**Code pt 4 — Training and evaluation:** Loop over episodes: reset env, run until done (collect transitions, update Q). Optionally log learning curves. Then evaluate: reset on validation data, run greedy, report metrics.

---

## Discussion

- **Overfitting:** Training on historical data can overfit. Use simple features, regularization, or validation-based early stopping. Do not expect the same performance on truly out-of-sample data.
- **Reward design:** Profit alone can lead to risky behavior. Consider risk-adjusted rewards (e.g. penalize variance) or constraints (position limits).
- **Extensions:** Multiple assets, continuous action (e.g. position size), or more advanced algorithms (DQN, policy gradients) as in later volumes.

This project ties together [Q-learning](../curriculum/volume-02/chapter-04/), [function approximation](../curriculum/volume-03/chapter-01/), and environment design. For more theory, see the [Course outline](../course-outline/) and [Curriculum](../curriculum/).
