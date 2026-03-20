---
title: "Phase 6 Assessment: RL Foundations"
description: "10–15 questions on MDPs, Bellman, MC vs TD, SARSA vs Q-learning. Solutions included."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["assessment", "phase 6", "foundations", "MDP", "Bellman", "SARSA", "Q-learning", "solutions"]
keywords: ["phase 6 foundations", "MDP Bellman", "Monte Carlo TD", "SARSA Q-learning", "solutions", "RL foundations quiz"]
weight: 10
roadmap_icon: "gamepad"
roadmap_color: "rose"
roadmap_phase_label: "Phase 6 Quiz"
---

Use this quiz after completing [Volume 1](../curriculum/volume-01/) and [Volume 2](../curriculum/volume-02/) (or the [Phase 6 mini-project](../learning-path/phase-3/)). If you can answer at least 12 of 15 correctly, you are ready for [Phase 7](../learning-path/#phase-7--deep-rl) and [Volume 3](../curriculum/volume-03/).

---

### 1. RL framework

**Q:** Name the four main components of an RL system (agent, environment, and two more). What is a state?

{{< collapse summary="Answer" >}}
Agent, environment, **action**, **reward**. **State:** a representation of the current situation the agent uses to choose actions.
{{< /collapse >}}

---

### 2. Return

**Q:** For rewards [0, 0, 1] and \\(\gamma = 0.9\\), compute the discounted return \\(G_0\\) from step 0.

{{< collapse summary="Answer" >}}
**Step 1:** Discounts are \\(\\gamma^0=1, \\gamma^1=0.9, \\gamma^2=0.81\\). **Step 2:** \\(G_0 = r_0 + \\gamma r_1 + \\gamma^2 r_2 = 0 + 0.9\\cdot 0 + 0.81\\cdot 1 = 0.81\\). **In RL:** The return from step 0 is the sum of discounted future rewards; we maximize this in every RL algorithm.
{{< /collapse >}}

---

### 3. Markov property

**Q:** What is the Markov property? Why is it important for planning?

{{< collapse summary="Answer" >}}
The future depends only on the current state and action, not on earlier history. It allows us to plan using only the current state (no need to remember the full history).
{{< /collapse >}}

---

### 4. Bellman equation

**Q:** Write the Bellman expectation equation for \\(V^\\pi(s)\\) in one line (in terms of \\(\\pi\\), \\(P\\), \\(r\\), \\(\\gamma\\), \\(V^\\pi\\)).

{{< collapse summary="Answer" >}}
\\(V^\\pi(s) = \\sum_a \\pi(a|s) \\sum_{s',r} P(s',r|s,a) [r + \\gamma V^\\pi(s')]\\).
{{< /collapse >}}

---

### 5. Discount factor

**Q:** What happens when \\(\gamma = 0\\)? When \\(\gamma = 1\\) in a continuing task (no terminal state)?

{{< collapse summary="Answer" >}}
\\(\gamma=0\\): agent is myopic (only immediate reward matters). \\(\gamma=1\\): future rewards weighted equally; in continuing tasks the return can be infinite unless we use average reward or other formulation.
{{< /collapse >}}

---

### 6. MC vs TD

**Q:** What is the key difference between Monte Carlo and TD learning in how they update the value estimate?

{{< collapse summary="Answer" >}}
MC uses the **full return** from that state to the end of the episode. TD uses **bootstrapping**: immediate reward plus the current estimate of the next state's value (no need to wait for episode end).
{{< /collapse >}}

---

### 7. SARSA vs Q-learning

**Q:** Write the Q-learning update for a transition \\((s, a, r, s')\\). How does the TD target differ from SARSA's target?

{{< collapse summary="Answer" >}}
\\(Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]\\). Q-learning uses \\(\\max_{a'} Q(s',a')\\); SARSA uses \\(Q(s', a')\\) where \\(a'\\) is the action actually taken in \\(s'\\) (on-policy).
{{< /collapse >}}

---

### 8. On-policy vs off-policy

**Q:** Is Q-learning on-policy or off-policy? Is SARSA? Explain in one sentence each.

{{< collapse summary="Answer" >}}
**Q-learning:** off-policy (learns about the greedy policy while following an exploratory policy, e.g. ε-greedy). **SARSA:** on-policy (learns about the policy that generates the actions, i.e. ε-greedy).
{{< /collapse >}}

---

### 9. Policy iteration

**Q:** Name the two steps of policy iteration. What do we do in each?

{{< collapse summary="Answer" >}}
**Policy evaluation:** compute \\(V^\\pi\\) for the current policy (iterate Bellman expectation until convergence). **Policy improvement:** make the policy greedy w.r.t. current V (or Q). Repeat until policy no longer changes.
{{< /collapse >}}

---

### 10. Value iteration

**Q:** How does value iteration differ from policy iteration? What update do we do in value iteration?

{{< collapse summary="Answer" >}}
Value iteration does not maintain an explicit policy; it iterates \\(V_{k+1}(s) = \\max_a \\sum_{s',r} P(s',r|s,a)[r + \\gamma V_k(s')]\\) until convergence, then derives the greedy policy from \\(V\\).
{{< /collapse >}}

---

### 11. First-visit MC

**Q:** In first-visit MC prediction, how many returns do we use per state per episode? What about every-visit MC?

{{< collapse summary="Answer" >}}
**First-visit:** at most one return per state per episode (the return from the first time we visit that state). **Every-visit:** we use the return from every time we visit that state in the episode.
{{< /collapse >}}

---

### 12. Function approximation

**Q:** Why is function approximation needed for large or continuous state spaces?

{{< collapse summary="Answer" >}}
Tabular methods store one value per state (or state-action); the number of states can be huge or infinite, so we cannot store or visit them all. Function approximation uses a parameterized function (e.g. linear or neural network) so a fixed number of parameters represent values for all states and generalize from seen to unseen states.
{{< /collapse >}}

---

### 13. Exploration

**Q:** Give one example of an exploration strategy used in tabular RL. Why is exploration necessary?

{{< collapse summary="Answer" >}}
ε-greedy: with probability ε take a random action. Exploration is necessary so we try all actions and learn their values; otherwise we might stick to a suboptimal action forever.
{{< /collapse >}}

---

### 14. Dyna-Q

**Q:** In Dyna-Q, what is the "model"? How does planning with the model help sample efficiency?

{{< collapse summary="Answer" >}}
The model is a representation of the environment (e.g. (s,a) → (s', r)). We can simulate transitions from the model and perform Q-updates on them without taking real env steps, so we get more learning per real step.
{{< /collapse >}}

---

### 15. Scaling

**Q:** For a 10×10 grid with 4 actions, how many entries does a tabular Q-table have? Why is this a problem for a 100×100 grid?

{{< collapse summary="Answer" >}}
**Step 1:** Q-table size = states × actions = 100 × 4 = **400** entries. **Step 2:** For 100×100: 10,000 × 4 = 40,000 entries (still feasible but large). **Why it’s a problem:** For continuous or very large discrete spaces we have infinitely many or huge numbers of states; we cannot store or visit them all, so we need function approximation (e.g. neural net) with a fixed number of parameters.
{{< /collapse >}}

---

**Next step:** If you passed, go to [Phase 4 — Deep RL](../learning-path/#phase-4--deep-rl) and [Volume 3](../curriculum/volume-03/).
