---
title: "Chapter 70: Unsupervised Environment Design"
description: "Simple PAIRED: adversary designs maze, agent solves; train both."
date: 2026-03-10T00:00:00Z
weight: 70
draft: false
tags: ["UED", "PAIRED", "adversarial", "environment design", "curriculum"]
keywords: ["unsupervised environment design", "PAIRED", "adversary", "maze", "agent"]
---

**Learning objectives**

- **Implement** a simple PAIRED-style setup: an adversary that designs a maze (or environment) to minimize the agent's return, and an agent that learns to solve mazes.
- **Train** both adversary and agent in a loop: adversary proposes a maze, agent attempts to solve it, update adversary to make the maze harder and agent to improve.
- **Explain** how unsupervised environment design can produce a curriculum of tasks without hand-designed levels.
- **Compare** agent performance on adversary-generated mazes vs fixed or random mazes.
- **Relate** PAIRED to **game AI** (procedural level generation) and **robot navigation** (training on diverse scenarios).

**Concept and real-world RL**

**Unsupervised environment design (UED)** or **PAIRED**-style methods let an **adversary** generate environment parameters (e.g. maze layout, obstacle positions) to minimize the agent's return, while the **agent** tries to maximize return. The two are trained together: the adversary learns to design hard-but-solvable tasks, and the agent learns to solve whatever is thrown at it. This creates a **automatic curriculum** without hand-designed levels. In **game AI**, this relates to procedural level generation and robust play; in **robot navigation**, training on adversarially chosen scenarios can improve robustness to distribution shift.

**Where you see this in practice:** PAIRED and similar UED methods; procedural content generation for RL; robust policy learning.

**Illustration (PAIRED):** The adversary designs mazes to minimize agent return; the agent learns to solve harder mazes. The chart below shows agent return on a fixed test maze over training iterations.

{{< chart type="line" title="Agent return on fixed test maze (PAIRED)" labels="0, 100, 200, 300, 400" data="20, 50, 80, 120, 150" >}}

**Exercise:** Implement a simple version of PAIRED: an adversary designs a maze to minimize the agent's performance, and the agent learns to solve any maze. Train both adversary and agent in a loop.

**Professor's hints**

- **Maze parameterization:** e.g. binary grid (wall or free) of size H×W, or a small set of parameters (e.g. number of rooms, corridor length). The adversary outputs these parameters; the environment is built from them.
- **Adversary objective:** Maximize negative agent return (or minimize agent return). So gradient for adversary: increase loss when agent does well. Agent objective: maximize return. Alternate or joint updates: update agent on current maze(s), then update adversary to generate mazes where the agent does worse (but avoid unsolvable mazes if you want a curriculum).
- **Stability:** If the adversary makes mazes impossible, the agent gets no learning signal. Often the adversary is constrained (e.g. ensure at least one path to goal) or the objective is modified (e.g. "minimize return subject to solvability"). Start with a simple constraint (e.g. guarantee one path).
- Use a **small** maze (e.g. 5×5 or 7×7) and a discrete set of maze designs so that the adversary's output is tractable (e.g. small neural network or categorical distribution).

**Common pitfalls**

- **Adversary collapses:** The adversary might find a trivial way to "minimize" return (e.g. all walls, or unreachable goal). Add constraints or reward shaping so mazes remain valid and solvable.
- **Agent too weak early:** If the agent never solves any maze at the start, the adversary has no gradient. Consider starting with a pretrained agent or an initial phase where the adversary is weak (e.g. random mazes).
- **Balance:** The relative learning rates of adversary and agent matter; if the adversary improves too fast, the agent may never learn. Tune or use a curriculum (e.g. limit adversary strength initially).

{{< collapse summary="Worked solution (warm-up: adversarial RL)" >}}
**Key idea:** In adversarial RL we have two agents (or an agent and an adversary). The agent tries to maximize return; the adversary tries to minimize it (or to make the task hard). We train both; the equilibrium (if it exists) can yield a robust or diverse policy. Balance is key: curriculum (e.g. weak adversary first) or separate learning rates so the agent can keep up. Used in robust RL and some multi-agent settings.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why might training an agent on adversarially chosen mazes lead to more robust behavior than training on fixed or random mazes?
2. **Coding:** Implement a minimal PAIRED: 5×5 grid, adversary outputs a 5×5 binary wall grid (with constraints: start at (0,0), goal at (4,4), at least one path). Agent is a small policy (e.g. MLP). Train for 500 iterations: agent does 10 episodes on current maze, adversary updates to reduce agent return. Plot agent return on a fixed test maze over training.
3. **Challenge:** Add **diversity** to the adversary: encourage it to generate mazes that are different from each other (e.g. diversity bonus or entropy regularizer on maze distribution). Compare the diversity of mazes and agent generalization with and without diversity.
