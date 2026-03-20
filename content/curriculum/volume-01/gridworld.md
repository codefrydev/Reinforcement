---
title: "Gridworld"
description: "The classic gridworld environment: states, actions, transitions, and terminal states."
date: 2026-03-10T00:00:00Z
weight: 4
draft: false
difficulty: 6
tags: ["gridworld", "MDP", "environment", "curriculum"]
keywords: ["gridworld", "MDP", "reinforcement learning", "environment"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Gridworld"
---

**Learning objectives**

- Define a gridworld MDP: grid cells as states, actions (up/down/left/right), transitions, and terminal states.
- Understand how hitting the boundary keeps the agent in place (or wraps, depending on design).
- Use gridworld as the running example for policy evaluation and policy iteration.

## What is Gridworld?

**Gridworld** is a simple MDP used throughout RL teaching and research. The environment is a grid of cells (e.g. 4×4 or 5×5). The **state** is the agent’s position \\((i, j)\\). **Actions** are typically up, down, left, right. **Transitions:** taking an action moves the agent one cell in that direction; if the move would go off the grid, the agent either stays in place (and usually receives the same step reward) or the world wraps, depending on the specification.

**Terminal states** are special cells where the episode ends. Common setups:

- **4×4 Gridworld (Sutton & Barto):** Two terminal states, e.g. top-left (0,0) and bottom-right (3,3). Every *non-terminal* step gives reward \\(-1\\). Terminal states give 0 reward and no further transitions. The goal is to reach a terminal state quickly (minimize negative reward = minimize steps).

- **Obstacles:** Some cells can be blocked; moving into them leaves the agent in the current cell (or the move is invalid). Rewards can be placed on specific cells (e.g. +10 at goal, -10 at pit).

## Why use Gridworld?

- **Small and discrete:** Easy to enumerate states and actions, so we can implement exact dynamic programming (policy evaluation, value iteration).
- **Visual:** Value functions and policies can be drawn as heatmaps and arrows.
- **Extensible:** Add wind (see [Windy Gridworld](windy-gridworld/)), stochastic transitions, or larger grids for later topics.

## In code

States are often represented as tuples \\((row, col)\\) or a single index \\(i = row \times \text{cols} + col\\). Actions can be 0=up, 1=down, 2=left, 3=right (or N/S/E/W). The transition function: given state \\(s\\) and action \\(a\\), compute the next cell; if out of bounds, next cell = current cell. If next cell is terminal, the transition returns that state and the episode is done.

See [Choosing Rewards](choosing-rewards/) for how to set rewards, [Chapter 7: Policy Evaluation](chapter-07/) for iterative policy evaluation on gridworld, and [Gridworld in Code](dp-gridworld-in-code/) for a code walkthrough.
