---
title: "Windy Gridworld"
description: "Gridworld with wind: actions are shifted by a wind effect. Theory and code for policy evaluation and policy iteration."
date: 2026-03-10T00:00:00Z
weight: 8
draft: false
difficulty: 6
tags: ["windy gridworld", "MDP", "dynamic programming", "curriculum"]
keywords: ["windy gridworld", "Sutton and Barto", "gridworld", "wind"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Windy Gridworld"
---

**Learning objectives**

- Understand the Windy Gridworld environment: movement is shifted by a column-dependent wind.
- Implement the transition model and run iterative policy evaluation and policy iteration on it.
- Compare with the standard gridworld (no wind).

## Theory

**Windy Gridworld** (Sutton & Barto) is a rectangular grid (e.g. 7×10) with:

- **States:** Cell positions \\((row, col)\\).
- **Actions:** Up, down, left, right (four actions).
- **Wind:** Each column has a fixed wind strength (non-negative integer). When the agent takes an action, the *resulting* row is shifted *up* by the wind strength (wind blows upward). So from cell \\((r, c)\\), after applying action “up” you might move to \\((r - 1 + \text{wind}[c], c)\\); “down” gives \\((r + 1 + \text{wind}[c], c)\\), etc. The agent cannot go below row 0 or above the grid; positions are clipped to the grid.
- **Terminal state:** One goal cell. Typical reward: -1 per step until the goal.

So the same action can lead to different next states depending on the column (wind). The MDP is still finite and deterministic given state and action (wind is fixed per column). This makes the problem slightly harder than a plain gridworld and is a good testbed for policy evaluation and policy iteration.

## Iterative policy evaluation for Windy Gridworld

Same as for standard gridworld: define \\(P(s', r | s, a)\\) from the wind rule (one next state per \\((s,a)\\) because the environment is deterministic). Apply the Bellman expectation update for the given policy until \\(V\\) converges. Terminal state has value 0 and is not updated.

## Policy iteration in Windy Gridworld

After computing \\(V^\\pi\\), improve the policy greedily: in each state, choose the action that maximizes immediate reward plus \\(\gamma V(s')\\). Then re-evaluate the new policy. Repeat until the policy does not change. The optimal policy will account for wind (e.g. moving “down” in a column with strong wind might be needed to reach the goal).

## Code

- Build the transition function: for each \\((s, a)\\), compute the next cell (action effect + wind, then clip to grid). If next cell is terminal, transition is done.
- Use the same iterative policy evaluation and policy iteration loops as in [Gridworld in Code](dp-gridworld-in-code/), with this transition function.
- Optional: visualize the value function and the greedy policy (arrows) on the grid.

See [Gridworld](gridworld/) and [Chapter 7: Policy Evaluation](chapter-07/) for the non-windy case, and [DP Gridworld in Code](dp-gridworld-in-code/) for a code walkthrough.
