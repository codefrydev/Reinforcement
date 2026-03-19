---
title: "Checkpoint: Volume 1, Midpoint (After Chapter 5)"
description: "5 quick questions after Chapters 1–5 of Volume 1. Check you're ready to continue."
date: 2026-03-19T00:00:00Z
draft: false
tags: ["checkpoint", "volume 1", "assessment", "value functions", "Bellman"]
---

Take this checkpoint after completing Chapters 1–5 (RL framework, bandits, MDPs, reward hypothesis, value functions). All 5 should feel manageable — if any are unclear, re-read the relevant chapter before continuing.

---

**Q1.** Name the five components of an MDP. Write the tuple.

{{< collapse summary="Answer" >}}
**(S, A, P, R, γ)** — State space, Action space, Transition probabilities, Reward function, Discount factor.
{{< /collapse >}}

---

**Q2.** In a gridworld, the agent is at (2,1) and moves right to (2,2) which is the goal. What is: (a) the state before the action, (b) the action, (c) the next state, (d) the reward (assuming +1 at goal, -1 per step)?

{{< collapse summary="Answer" >}}
(a) state s = **(2,1)**, (b) action a = **right**, (c) next state s' = **(2,2)**, (d) reward r = **+1** (goal reached).
{{< /collapse >}}

---

**Q3.** The discount factor γ = 0.9. A reward of +1 arrives after 3 steps. What is its present value?

{{< collapse summary="Answer" >}}
γ³ × 1 = 0.9³ = **0.729**.
{{< /collapse >}}

---

**Q4.** V^π(s) is defined as "the expected discounted return from state s, following policy π." Write the Bellman expectation equation for V^π(s) in words (no need for full notation).

{{< collapse summary="Answer" >}}
V^π(s) = **the expected immediate reward (averaged over actions and transitions) plus γ times the expected value of the next state, averaged over the same actions and transitions under policy π**.

Formally: V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γ V^π(s')].
{{< /collapse >}}

---

**Q5.** What is the difference between V(s) and Q(s,a)?

{{< collapse summary="Answer" >}}
- **V(s)**: value of state s — expected return starting from s and following policy π.
- **Q(s,a)**: value of taking action a in state s — expected return after taking action a in state s, then following policy π.

Relationship: V(s) = Σ_a π(a|s) Q(s,a) (V averages Q over actions under the policy).
{{< /collapse >}}

---

All 5 correct? Continue to Chapter 6 (Bellman Equations). Stuck on 2 or more? Re-read Chapters 3–5.
