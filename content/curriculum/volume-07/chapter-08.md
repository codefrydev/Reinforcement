---
title: "Chapter 68: Model-Agnostic Meta-Learning (MAML) in RL"
description: "MAML for locomotion (e.g. different velocities); one-step adapt."
date: 2026-03-10T00:00:00Z
weight: 68
draft: false
tags: ["MAML", "meta-learning", "locomotion", "adaptation", "curriculum"]
keywords: ["MAML", "Model-Agnostic Meta-Learning", "one-step adapt", "locomotion"]
---

**Learning objectives**

- **Implement** MAML for a simple RL task: sample tasks (e.g. different target velocities), compute inner update (one or a few gradient steps on task loss), then meta-update using the post-adaptation loss.
- **Compute** the meta-gradient (gradient of the post-adaptation return or loss w.r.t. initial parameters), using second-order derivatives or a first-order approximation.
- **Explain** why MAML learns an initialization that is "easy to fine-tune" with one or few gradient steps.
- **Train** a policy that adapts in one gradient step to a new task and evaluate on held-out tasks.
- **Relate** MAML to **robot navigation** (e.g. different terrains or payloads) and **game AI** (different levels).

**Concept and real-world RL**

**Model-Agnostic Meta-Learning (MAML)** finds an **initialization** of the policy (or value function) such that after **one or a few gradient steps** on a new task, performance is high. In RL, each task might be a different reward function (e.g. different target velocity in locomotion) or different dynamics. The meta-objective is the return (or loss) after the inner update; the meta-gradient requires differentiating through the inner gradient step. In **robot navigation** and **game AI**, this enables fast adaptation to new goals, terrains, or opponents with minimal data. First-order MAML (FOMAML) omits second-order terms for simplicity and is often used in practice.

**Where you see this in practice:** MAML and Reptile for few-shot RL; meta-learning for robotics and control.

**Illustration (MAML adaptation):** After one gradient step on the new task, the adapted policy often achieves much higher return than the initial policy. The chart below shows return before and after 1 inner step.

{{< chart type="bar" title="Return: before vs after 1 adaptation step" labels="Before, After 1 step" data="30, 85" >}}

**Exercise:** Implement MAML for a simple locomotion task (e.g., different velocities). Train a policy that can adapt in one gradient step. Compute the meta-gradient using second-order derivatives (or first-order approximation).

**Professor's hints**

- **Tasks:** e.g. HalfCheetah or a simple 1D/2D locomotion with different target velocities; reward = negative squared error to target velocity. Sample a new target velocity each task.
- **Inner step:** On task \\(\\tau\\), compute loss \\(L_\\tau(\\theta)\\) (e.g. negative return), then \\(\\theta' = \\theta - \\alpha \\nabla_\\theta L_\\tau(\\theta)\\). Use a small \\(\\alpha\\) (e.g. 0.01).
- **Meta-update:** Meta-loss = \\(L_\\tau(\\theta')\\) (return on same or new rollouts with \\(\\theta'\\)). Meta-gradient = \\(\\nabla_\\theta L_\\tau(\\theta')\\); this requires backprop through \\(\\theta'\\). For first-order approximation, use \\(\\nabla_{\\theta'} L_\\tau(\\theta')\\) and treat \\(\\theta' \\approx \\theta\\) for the gradient (no second-order terms).
- Use a small network and short rollouts (e.g. 20 steps per task) so that inner and outer updates are tractable.

**Common pitfalls**

- **Second-order cost:** Full MAML needs Hessian-vector products or double backprop; use first-order MAML or a small number of inner steps to reduce compute.
- **Inner learning rate:** If \\(\\alpha\\) is too large, the inner update may overshoot; if too small, adaptation is weak. Tune \\(\\alpha\\) and consider per-parameter or learned inner LR.
- **Variance:** Meta-gradient can have high variance; use multiple tasks per meta-update and average the meta-gradient.

{{< collapse summary="Worked solution (warm-up: meta-gradient RL)" >}}
**Key idea:** Meta-gradient methods tune hyperparameters (e.g. \\(\\lambda\\) in TD(\\(\\lambda\\)), or step size) by gradient descent on a meta-objective (e.g. return or validation loss). We need to differentiate through the learning process; this can be first-order (treat inner update as fixed) or second-order (through the inner update). The meta-parameters are updated so that the agent learns better on future tasks or episodes.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence, what does the MAML meta-gradient optimize? (The loss after one gradient step. So we want initial parameters such that one step on any task reduces the loss a lot.)
2. **Coding:** Implement first-order MAML on a 2D point-mass task: goal is to move to a target position; each task = different target. Inner: one policy gradient step. Outer: gradient of post-adaptation return. Train for 200 meta-iterations and plot return on held-out tasks after one inner step.
3. **Challenge:** Implement **full** second-order MAML (backprop through the inner update). Compare meta-training time and final adaptation performance with first-order MAML.
