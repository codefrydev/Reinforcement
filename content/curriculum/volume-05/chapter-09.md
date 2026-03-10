---
title: "Chapter 49: Custom Gym Environments (Part 2)"
description: "Custom 2D point mass with continuous action; test with SAC."
date: 2026-03-10T00:00:00Z
weight: 49
draft: false
tags: ["custom Gym", "continuous action", "point mass", "SAC", "curriculum"]
keywords: ["custom Gym environment", "continuous action", "point mass", "SAC"]
---

**Learning objectives**

- Create a **custom Gym environment**: a **2D point mass** that must navigate to a goal while avoiding an obstacle.
- Define **continuous action** (e.g. force in x and y) and a **reward function** (e.g. distance to goal, penalty for obstacle or boundary).
- **Test** the environment with a SAC (or PPO) agent and verify that the agent can learn to reach the goal.

**Concept and real-world RL**

Custom environments let you model **robot navigation**, **recommendation** (state = user, action = item), or **trading** (state = market, action = trade). A 2D point mass is a minimal continuous control task: state = (x, y, vx, vy), action = (fx, fy), reward = -distance to goal + penalties. In **robot control**, similar point-mass or particle models are used for planning and RL; in **game AI**, custom envs are used for prototyping. Implementing the Gym interface (reset, step, observation_space, action_space) and testing with a known algorithm (SAC) validates the design.

**Where you see this in practice:** Research and industry often use custom Gym envs for domain-specific problems (warehouse robots, driving, dialogue).

**Illustration (2D point mass state):** A simple continuous control env might have state (x, y, vx, vy) and action (fx, fy). The chart below shows a trajectory in (x,y) over 20 steps (conceptual scatter).

{{< chart type="scatter" title="(x,y) trajectory (point mass to goal)" data="0,0 0.5,0.2 1,0.5 1.2,0.8 1.5,1 1.8,1.2 2,1.5" >}}

**Exercise:** Create a custom continuous control environment: a 2D point mass that must navigate to a goal while avoiding an obstacle. Define a continuous action (force) and a reward function. Test your environment with a SAC agent.

**Professor's hints**

- Subclass `gym.Env`; implement `reset()` (return obs, info) and `step(action)` (return obs, reward, terminated, truncated, info). Set `observation_space` (e.g. Box(4,) for x,y,vx,vy) and `action_space` (Box(2,) for fx, fy, bounded).
- Dynamics: e.g. \\(x_{t+1} = x_t + v_x dt\\), \\(v_{x,t+1} = v_{x,t} + f_x dt\\) (with clipping). Goal at (1,1), obstacle as a circle; reward = -distance_to_goal - 10 if in obstacle, or sparse (+1 at goal).
- Test: run SAC for 50k steps; plot position (x,y) over time. Does the agent eventually reach the goal?

**Common pitfalls**

- **Reward shaping:** Too much shaping can make the agent exploit loopholes; too sparse can make learning slow. Start simple (-distance) and add obstacle penalty.
- **Action scale:** Clip or scale actions to a reasonable force; otherwise the point mass can shoot off.

{{< collapse summary="Worked solution (warm-up: continuous control)" >}}
**Key idea:** In continuous control we output a distribution over actions (e.g. Gaussian with mean from the network and learned or fixed std). We sample \\(a \\sim \\pi(\\cdot|s)\\), compute \\(\\nabla \\log \\pi(a|s)\\), and use it with the advantage (e.g. TD error or GAE). For bounded action spaces we squash through tanh and add the log-Jacobian to the log-probability. SAC and PPO both support continuous actions this way.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** What should the observation space and action space be for a 2D point mass? (State: position and velocity; action: force.)
2. **Coding:** Implement the env and run random actions for 100 steps. Check that (x,y) moves and that reward is computed. Then run SAC for 20k steps and plot the trajectory of (x,y) for one episode.
3. **Challenge:** Add a **moving obstacle** (e.g. oscillating circle). Does SAC still learn to reach the goal while avoiding it?
