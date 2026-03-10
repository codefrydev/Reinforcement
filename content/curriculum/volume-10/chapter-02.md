---
title: "Chapter 92: Safe Reinforcement Learning"
description: "Constrained MDP for self-driving; Lagrangian penalty."
date: 2026-03-10T00:00:00Z
weight: 92
draft: false
---

**Learning objectives**

- **Formulate** a **constrained MDP** for a self-driving car (or similar): maximize progress (or reward) while keeping **collisions** (or another cost) **below a threshold**.
- **Implement** a **Lagrangian** method: add a penalty term λ * (constraint violation) to the objective and update the **penalty coefficient** λ (e.g. increase λ when the constraint is violated) so that the policy satisfies the constraint.
- **Explain** the trade-off: higher λ pushes the policy to satisfy the constraint but may reduce task reward; tune λ or use dual ascent.
- **Evaluate** the policy: report task return and constraint cost (e.g. number of collisions per episode); verify the constraint is met in evaluation.
- **Relate** safe RL and constrained MDPs to **healthcare** (safety constraints) and **trading** (risk limits).

**Concept and real-world RL**

**Safe RL** aims to maximize reward while satisfying **constraints** (e.g. collision probability below a threshold, or expected cost below a bound). A **constrained MDP** formulates this as: maximize E[return] subject to E[cost] ≤ d. The **Lagrangian** approach turns this into an unconstrained problem: maximize E[return] - λ * (E[cost] - d) or similar, and **update λ** (e.g. increase if constraint violated) so that at convergence the constraint is satisfied. In **self-driving**, **healthcare**, and **trading**, constraints (collisions, harm, risk) are critical; Lagrangian and other safe-RL methods are used to enforce them.

**Where you see this in practice:** CPO, PPO-Lagrangian, safe RL benchmarks; constrained MDPs in control and robotics.

**Exercise:** Formulate a constrained MDP for a self-driving car: maximize progress while keeping collisions below a threshold. Implement a Lagrangian method that updates a penalty coefficient to enforce the constraint.

**Professor's hints**

- **MDP:** State = position, velocity, other cars; action = steer, accelerate/brake; reward = progress (e.g. +1 per step or speed); cost = 1 if collision, 0 otherwise. Constraint: E[sum of cost per episode] ≤ d (e.g. d = 0.01, so at most 1% of episodes have a collision, or expected collisions per episode ≤ 0.01).
- **Lagrangian:** Optimize policy to maximize (return - λ * (cost - d)). So we penalize cost above d. Update λ: e.g. λ += α * (average_cost - d) each epoch, so if we violate the constraint, λ increases and the next policy will care more about cost.
- **Implementation:** Use PPO or similar; the "reward" in the PPO update is (r - λ * c) where c is the cost at that step (or discounted cost). Run for many episodes; each N episodes, update λ based on average constraint violation.
- Use a **simple** sim (e.g. 1D or 2D car, or a grid with obstacles) so you can focus on the Lagrangian logic.

**Common pitfalls**

- **λ too large:** If λ becomes huge, the policy may only minimize cost and ignore task reward. Clip λ or use a bounded dual update.
- **Constraint not measurable:** Ensure cost is well-defined (e.g. binary collision per step) and that you can estimate E[cost] from rollouts.
- **Delayed cost:** If cost (e.g. collision) is rare, the gradient may be high variance; use many episodes to estimate the constraint before updating λ.

**Extra practice**

1. **Warm-up:** In the Lagrangian formulation, what happens to the penalty coefficient λ when the constraint is violated (E[cost] > d)? Why does that help?
2. **Coding:** Implement a 1D or 2D driving env: reward = speed, cost = 1 if collision. Constraint: E[cost per episode] ≤ 0.1. Train with PPO and Lagrangian (update λ every 10 episodes). Plot return, cost, and λ over training. Does cost stay below the threshold?
3. **Challenge:** Implement **CPO** or **projection-based** constrained update: after each PPO step, project the policy onto the constraint satisfaction set (e.g. one more gradient step to reduce cost). Compare with Lagrangian in terms of constraint satisfaction and return.
