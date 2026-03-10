---
title: "Chapter 18: Custom Gym Environments (Part 1)"
description: "Custom 2D maze Gym env with text render."
date: 2026-03-10T00:00:00Z
weight: 18
draft: false
tags: ["Gym", "custom environment", "maze", "curriculum"]
keywords: ["custom Gym environment", "OpenAI Gym", "environment API", "maze"]
---

**Learning objectives**

- Create a custom Gymnasium (or Gym) environment: inherit from `gym.Env`, implement `reset`, `step`, and optional `render`.
- Define `observation_space` and `action_space` (e.g. Discrete(4) for up/down/left/right).
- Implement a text-based render (e.g. print a grid with agent and goal).

**Concept and real-world RL**

Real RL often requires **custom environments**: simulators for robotics, games, or domain-specific tasks. The Gym API (`reset`, `step`, `observation_space`, `action_space`) is the standard. Implementing a small maze teaches you how to encode state (e.g. agent position), handle boundaries and obstacles, and return (obs, reward, terminated, truncated, info). In practice, you will wrap or write envs for your problem and reuse the same agents (e.g. Q-learning, DQN) trained on standard envs.

**Illustration (tabular Q size):** For a 5×5 maze with 4 actions, the Q-table has states × actions entries. With 23 free cells (e.g. 2 walls), that is 23 × 4 = 92 entries. The chart below shows how Q-table size grows with maze size (states × 4 actions).

{{< chart type="bar" title="Q-table entries (states × 4 actions)" labels="3×3, 5×5, 7×7" data="36, 100, 196" >}}

**Exercise:** Create a custom Gym environment for a 2D maze with obstacles. Define observation (agent position) and discrete actions (up, down, left, right). Implement a render function that prints a text-based map.

**Professor's hints**

- Subclass `gymnasium.Env`. In `__init__`, set `self.observation_space` (e.g. `gymnasium.spaces.Box` for position or `Discrete` for cell index) and `self.action_space = gymnasium.spaces.Discrete(4)`.
- `reset(seed=None)`: set agent to start position, return `(obs, info)`. Obs can be a tuple `(row, col)` or a flattened index; make it hashable or numpy for compatibility.
- `step(action)`: compute next position (clip to grid, handle walls). If next cell is obstacle, stay and maybe give negative reward. If goal, `terminated=True`, positive reward. Return `(obs, reward, terminated, truncated, info)`.
- `render()`: build a 2D grid of characters (e.g. '#' wall, '.' empty, 'A' agent, 'G' goal), print row by row. Use `mode="human"` or similar if required by the API.

**Common pitfalls**

- **Gymnasium vs Gym:** Gymnasium uses `terminated` and `truncated`; Gym (old) used a single `done`. Use both flags and set `done = terminated or truncated` in your training loop.
- **Observation type:** Many algorithms expect a numpy array or a consistent type. Avoid returning a different shape in different states (e.g. terminal). Use a fixed obs space even for terminal (e.g. same shape, zero or last state).
- **Action semantics:** Document whether 0=up, 1=down, etc., and be consistent. In a 2D grid, "up" often means decrease row index.

{{< collapse summary="Worked solution (warm-up: maze Q-table size)" >}}
**Warm-up:** In your maze, how many possible observations (states)? How many actions? Size of tabular Q-table? **Answer:** States = number of cells the agent can occupy (e.g. 5×5 minus walls ⇒ e.g. 23 if 2 walls). Actions = 4 (up, down, left, right). Q-table size = states × actions (e.g. 23 × 4 = 92 entries). So even a small maze has a small table; the same idea scales to larger grids or discrete state spaces.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In your maze, how many possible observations (states) are there? How many actions? What is the size of a tabular Q-table?
2. **Coding:** Implement a minimal Gym maze (e.g. 5×5, start (0,0), goal (4,4), walls optional). Implement reset() and step(action). Run a random policy for 10 episodes and log episode length and return.
3. **Challenge:** Add a "time limit" (e.g. 100 steps): set `truncated=True` when step count exceeds 100. Return this in `info` as well. Run a random agent for one episode and confirm truncation occurs.
