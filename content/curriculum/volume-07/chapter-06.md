---
title: "Chapter 66: Go-Explore Algorithm"
description: "Simplified Go-Explore on deterministic maze; archive and return."
date: 2026-03-10T00:00:00Z
weight: 66
draft: false
tags: ["Go-Explore", "archive", "exploration", "maze", "curriculum"]
keywords: ["Go-Explore", "archive", "deterministic maze", "exploration"]
---

**Learning objectives**

- **Implement** a simplified Go-Explore: an archive of promising states and a strategy to return to them and explore further.
- **Explain** the two-phase idea: (1) archive states that lead to high rewards or novelty, (2) select from the archive, return to that state, then take exploratory actions.
- **Compare** Go-Explore with random exploration (e.g. episodes to reach goal, or maximum reward reached) on a deterministic maze.
- **Identify** why "return" (resetting to an archived state) helps in hard exploration compared to always starting from the initial state.
- **Relate** Go-Explore to **game AI** (e.g. Montezuma's Revenge) and **robot navigation** with sparse goals.

**Concept and real-world RL**

**Go-Explore** addresses hard exploration by explicitly **archiving** promising states (e.g. states that achieved a high reward or are novel) and then **returning** to them to explore further, rather than always starting from the initial state. In a deterministic environment, the agent can reset to an archived state and try new actions from there, so it can reach distant goals by building a chain of archived waypoints. In **game AI** (e.g. Montezuma's Revenge), this has led to strong results; in **robot navigation**, the idea of "save checkpoints and explore from them" mirrors classical exploration strategies. A simplified version uses a deterministic maze, an archive (e.g. by state or by cell), and a policy to select which archived state to return to and how to explore.

**Where you see this in practice:** Go-Explore on Atari hard-exploration games; similar "reset to good state" ideas in robotics and planning.

**Illustration (Go-Explore archive):** The archive stores promising states; the agent returns to them to explore further. The chart below shows the number of archive states and max return over iterations.

{{< chart type="line" palette="return" title="Archive size and max return" labels="0, 20, 40, 60, 80" data="5, 25, 60, 120, 200" xLabel="Iteration" yLabel="Archive size / Max return" >}}

**Exercise:** Implement a simplified Go-Explore on a deterministic maze: archive states that lead to high rewards, then return to them and explore further. Compare with random exploration.

**Professor's hints**

- **Archive:** A set or list of states (e.g. (row, col) in a grid). Add a state to the archive when it reaches a new high reward or is "interesting" (e.g. first time visiting that cell with a high return so far).
- **Selection:** Each episode, select a state from the archive (e.g. uniformly at random, or by priority such as reward or recency). **Return:** Start the episode from that state (deterministic env allows this); if your env does not support arbitrary start, simulate a trajectory from the start that reaches that state and then continue.
- **Explore:** From the selected state, take many random or exploratory actions and see if you reach new states; add new promising states to the archive.
- Use a **deterministic** maze so that "return to state S" is well-defined. Compare with an agent that always starts from the initial state and uses random or ε-greedy exploration; measure episodes or steps to reach the goal.

**Common pitfalls**

- **Stochastic environments:** Go-Explore assumes you can reset to an exact state; in stochastic envs you need a different formulation (e.g. robust or cell-based). Stick to deterministic for the simplified version.
- **Archive explosion:** If you archive every state, the archive becomes huge and selection is slow; use criteria (e.g. only states that improved reward, or one state per cell) to keep the archive manageable.
- **Forgetting the goal:** The explore phase should still reward or track progress toward the goal; otherwise the agent may only archive states that are easy to reach but not on the path to the goal.

{{< collapse summary="Worked solution (warm-up: goal-conditioned exploration)" >}}
**Key idea:** In goal-conditioned RL we train the agent to reach a set of goals. Exploration can be directed: e.g. sample a goal (from a curriculum or from an archive of reached states), then try to reach it. The agent gets reward for reaching the chosen goal. This focuses exploration on "reachable but not yet mastered" goals and can be much more efficient than undirected exploration in sparse-reward settings.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In a 10×10 maze with the goal at (9,9), why might "return to (5,5) and explore" lead to finding the goal faster than always starting at (0,0)?
2. **Coding:** Implement a minimal Go-Explore: archive = set of (row, col) reached with reward \\(\\ge\\) threshold. Each iteration, pick a random archived state, reset to it, take 20 random steps, and add any new state that gets reward \\(\\ge\\) threshold. Run for 500 iterations and count how often you reach the goal vs random exploration from (0,0).
3. **Challenge:** Add **prioritized** selection: prefer archived states that have been visited fewer times or that led to higher reward. Compare archive size and time to goal with uniform selection.
