---
title: "Chapter 1: The Reinforcement Learning Framework"
description: "Gridworld discounted return from a sequence of actions."
date: 2026-03-10T00:00:00Z
weight: 1
draft: false
difficulty: 6
tags: ["reinforcement learning", "gridworld", "discounted return", "MDP", "curriculum"]
keywords: ["RL framework", "gridworld", "discounted return", "agent environment", "reward"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Ch 1"
---

{{< notebook path="volume-01/ch01_gridworld_return.ipynb" title="Open Chapter 1 notebook" >}}

**Learning objectives**

- Identify the main components of an RL system: agent, environment, state, action, reward.
- Compute the discounted return for a sequence of rewards.
- Relate the gridworld to real tasks (e.g. navigation, games) where an agent gets delayed reward.

**Concept and real-world RL**

In reinforcement learning, an **agent** interacts with an **environment**: at each step the agent is in a **state**, chooses an **action**, and receives a **reward** and a new state. The **return** is the sum of (discounted) rewards along a trajectory; the agent’s goal is to maximize this return. A **gridworld** is a simple environment where states are cells and actions move the agent; it models **robot navigation** (e.g. a robot moving to a goal in a warehouse) and **game AI** (e.g. a character moving on a map). In robot navigation, the state might be (row, col); the action is up/down/left/right; the reward is +1 at the goal and often 0 or a small penalty per step. Discounting (\\(\gamma < 1\\)) makes future rewards worth less than immediate ones and keeps the return finite in long or infinite horizons.

**Where you see this in practice:** Gridworld-style MDPs appear in pathfinding, warehouse robots, and simple games. The same agent–environment–return framework is used in recommendation (state = user context, action = which item to show, return = long-term engagement).

**Illustration (discounted return):** For rewards \\([0, 0, 1]\\) and \\(\gamma = 0.9\\), the return from step 0 is \\(0.81\\). The chart below shows how the return from step 0 changes as we add more future rewards (e.g. for a longer sequence). With only the first three rewards we get 0.81; if the episode continued with more zeros and a final 1, the pattern would extend similarly.

{{< chart type="line" palette="return" title="Cumulative discounted return (γ=0.9)" labels="1 step, 2 steps, 3 steps" data="0, 0, 0.81" xLabel="Step" yLabel="Return" >}}

**Agent-environment interaction loop:**

{{< mermaid >}}
flowchart LR
    Agent -->|"action a_t"| Environment
    Environment -->|"reward r_t"| Agent
    Environment -->|"next state s_{t+1}"| Agent
    Agent -->|"observes state s_t"| Agent
{{< /mermaid >}}

**Exercise:** In a 3×3 gridworld, the agent starts at (0,0) and aims to reach a goal at (2,2) with a reward of +1. Every other step gives 0 reward, and hitting a wall (outside grid) gives -1 and stays in place. Write a Python function that takes a sequence of actions (up, down, left, right) and returns the total discounted return (\\(\gamma = 0.9\\)).

{{< pyrepl code="def discounted_return(rewards, gamma=0.9):\n    # TODO: return sum(gamma**t * r for t, r in enumerate(rewards))\n    pass\n\nprint(discounted_return([0, 0, 1], 0.9))   # expected: 0.81" height="220" >}}

**Professor's hints**

- Encode actions as you like (e.g. 0=up, 1=down, 2=left, 3=right) and update (row, col) accordingly. Moving "up" usually decreases the row index.
- Maintain current (row, col); for each action, compute the next cell. If the next cell is outside [0,2]×[0,2], stay in place and add reward -1; if it is (2,2), add +1 and you can stop (or continue; define whether the episode terminates at the goal).
- The return is \\(G = r_0 + \\gamma r_1 + \\gamma^2 r_2 + \\cdots\\). Use a loop: at each step add \\(\\gamma^t \\cdot r_t\\) to the total.
- Test with a short sequence that reaches (2,2) in a few steps and verify the return by hand.

**Common pitfalls**

- **Off-by-one in grid indices:** (0,0) is top-left if row 0 is "up"; check whether "up" means row-1 or row+1 and be consistent.
- **Forgetting to discount:** Each reward must be multiplied by \\(\gamma^t\\) where \\(t\\) is the step index (0, 1, 2, …). Do not sum raw rewards unless \\(\gamma=1\\).
- **Wall semantics:** Clarify whether "hitting a wall" gives -1 and the agent stays in the same cell, or whether the action is simply not applied; implement one convention consistently.

{{< collapse summary="Worked solution (warm-up: discounted return by hand)" >}}
**Warm-up:** For rewards \\([0, 0, 1]\\) and \\(\gamma = 0.9\\), compute \\(G_0\\) by hand. Then write a one-line loop that computes it in Python.

**Step 1 — By hand:** \\(G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 = 0 + 0.9 \cdot 0 + 0.81 \cdot 1 = 0.81\\).

**Step 2 — Python (one-line loop):**
```python
G = sum(0.9**t * r for t, r in enumerate([0, 0, 1]))
# G == 0.81
```
Or: `G = 0 + 0.9*0 + 0.81*1`. The same discounting structure is used in the gridworld: at each step add \\(\gamma^t \cdot r_t\\) to the total. This is the return from step 0; the agent’s goal is to maximize such returns.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For rewards \\([0, 0, 1]\\) and \\(\gamma = 0.9\\), compute \\(G_0\\) by hand. Then write a one-line loop that computes it in Python.
2. **Coding:** Write a function `discounted_return(rewards, gamma)` that returns \\(G_0\\) for a list of rewards. Test with rewards = [0, 0, 1], gamma = 0.9 (expected 0.81).
3. **Challenge:** Extend your function to support a **list of (state, reward)** pairs (e.g. from a saved trajectory) and compute the return from the first state. No environment logic—just the math.
4. **Variant:** Try \\(\gamma=0.5\\) and \\(\gamma=0.99\\) with the same reward sequence \\([0, 0, 1]\\). How does \\(G_0\\) change? For which \\(\gamma\\) does future reward matter most?
5. **Debug:** The code below has a bug — it sums raw rewards instead of discounting them. Find and fix it.

{{< pyrepl code="def buggy_return(rewards, gamma=0.9):\n    return sum(rewards)  # BUG: discount missing\n\n# Expected for [0, 0, 1]: 0.81\nprint(buggy_return([0, 0, 1]))  # prints 1, not 0.81\n\n# TODO: fix the function\ndef fixed_return(rewards, gamma=0.9):\n    pass\n\nprint(fixed_return([0, 0, 1]))  # should print 0.81" height="220" >}}

6. **Conceptual:** Explain why discounting (\\(\gamma < 1\\)) is useful even when the episode is finite. When would using \\(\gamma = 1\\) be problematic?
7. **Recall:** State the formula for \\(G_0\\) (discounted return from step 0) in terms of rewards \\(r_0, r_1, r_2, \ldots\\) and \\(\gamma\\) from memory.
