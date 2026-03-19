---
title: "Volume 2 Drills — Tabular Model-Free Methods"
description: "15 short drill problems for Volume 2: Monte Carlo, TD(0), SARSA, Q-learning, and n-step methods."
date: 2026-03-19T00:00:00Z
draft: false
weight: 99
tags: ["drills", "volume 2", "Monte Carlo", "TD", "SARSA", "Q-learning", "n-step", "practice"]
---

{{< notebook path="volume-02/vol02_drills.ipynb" title="Open drills notebook (interactive)" >}}

Short problems for Volume 2. Aim for under 5 minutes per problem. All solutions are in collapsible sections.

---

## Recall (R) — State definitions and rules

**R1.** Write the TD(0) update rule from memory. What are its three inputs?

{{< collapse summary="Answer" >}}
**TD(0) update:**

V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) − V(S_t)]

Three inputs: **(1)** the current state S_t, **(2)** the reward R_{t+1}, **(3)** the next state S_{t+1}.

The term in brackets is the **TD error** δ_t = R_{t+1} + γ V(S_{t+1}) − V(S_t). It measures how wrong the current estimate was.
{{< /collapse >}}

---

**R2.** What is the difference between on-policy and off-policy learning? Give one algorithm of each type.

{{< collapse summary="Answer" >}}
**On-policy:** the policy being *evaluated/improved* is the same policy used to *generate experience*. Example: **SARSA** — updates Q using the action actually taken by the current policy.

**Off-policy:** the policy being evaluated is *different* from the policy generating experience. The agent can learn about a target policy while following a different behavior policy. Example: **Q-learning** — updates Q using the greedy action (max), regardless of what action was actually taken.
{{< /collapse >}}

---

**R3.** What is the difference between first-visit and every-visit Monte Carlo?

{{< collapse summary="Answer" >}}
Both estimate V(s) by averaging returns, but they differ in which visits count:

- **First-visit MC:** for each episode, only the *first* visit to state s contributes a return to the average.
- **Every-visit MC:** *every* visit to state s in an episode contributes a return.

Both converge to V^π as the number of episodes → ∞. First-visit MC has unbiased estimates; every-visit MC has lower variance when states are visited many times per episode.
{{< /collapse >}}

---

**R4.** Why can Q-learning be riskier than SARSA in real-world (non-simulated) environments?

{{< collapse summary="Answer" >}}
Q-learning is **optimistic** — it learns the value of the *greedy* policy (max Q) even if the agent actually follows a more exploratory behavior policy. During training, the agent can stray into dangerous states while exploring, but Q-learning's updates assume it will behave optimally from those states.

**SARSA** learns the value of the policy it *actually follows* (including exploration). Near cliffs or dangerous states, SARSA will learn to avoid risky actions because it accounts for the exploratory moves that could lead to disaster. This is the classic "Cliff Walking" result: SARSA learns a safer path, Q-learning learns a shorter but risky path near the cliff edge.
{{< /collapse >}}

---

**R5.** What do n-step returns interpolate between? Write the 1-step and ∞-step special cases.

{{< collapse summary="Answer" >}}
n-step returns interpolate between **TD(0)** (1-step bootstrap) and **Monte Carlo** (full-episode return):

- **1-step (n=1):** G_t^{(1)} = R_{t+1} + γ V(S_{t+1}) — pure bootstrapping (TD target).
- **∞-step (n=∞):** G_t^{(∞)} = R_{t+1} + γ R_{t+2} + ... = actual return — pure Monte Carlo (no bootstrapping).
- **n-step:** G_t^{(n)} = R_{t+1} + γ R_{t+2} + ... + γ^{n-1} R_{t+n} + γ^n V(S_{t+n}).

**TD(λ)** takes a weighted average of all n-step returns using geometric weights (λ^{n-1}).
{{< /collapse >}}

---

## Compute (C) — Numerical exercises

**C1.** Apply a TD(0) update by hand: V(A) = 0.3, reward r = 0, V(B) = 0.5, α = 0.1, γ = 0.9. The agent transitions A → B. What is the new V(A)?

{{< pyrepl code="V_A = 0.3\nV_B = 0.5\nr = 0\nalpha = 0.1\ngamma = 0.9\n\ntd_error = r + gamma * V_B - V_A\nV_A_new = V_A + alpha * td_error\nprint(f'TD error  = {td_error:.4f}')   # 0 + 0.45 - 0.3 = 0.15\nprint(f'V_new(A)  = {V_A_new:.4f}')   # 0.3 + 0.1*0.15 = 0.315" height="200" >}}

{{< collapse summary="Answer" >}}
TD error δ = 0 + 0.9 × 0.5 − 0.3 = 0.45 − 0.3 = **0.15**.

V_new(A) = 0.3 + 0.1 × 0.15 = **0.315**.
{{< /collapse >}}

---

**C2.** For the same transition (A → B, r=0, γ=0.9), compare the SARSA and Q-learning update targets. Current Q-values: Q(A, left)=0.4, Q(A, right)=0.6, Q(B, left)=0.7, Q(B, right)=0.5. The agent was in A, took action "right", landed in B, and chose action "left" next.

{{< pyrepl code="r = 0\ngamma = 0.9\nQ = {'A': {'left': 0.4, 'right': 0.6},\n     'B': {'left': 0.7, 'right': 0.5}}\n\n# Action taken: A->right, next action chosen: B->left\nsarsa_target = r + gamma * Q['B']['left']      # uses actual next action\nql_target    = r + gamma * max(Q['B'].values()) # uses max over next actions\n\nprint(f'SARSA target   = {sarsa_target:.3f}')  # 0 + 0.9*0.7 = 0.63\nprint(f'Q-learn target = {ql_target:.3f}')     # 0 + 0.9*0.7 = 0.63 (same here!)\nprint('Same because left is also the greedy action in B.')" height="220" >}}

{{< collapse summary="Answer" >}}
SARSA target = 0 + 0.9 × Q(B, left) = 0 + 0.9 × 0.7 = **0.63**.

Q-learning target = 0 + 0.9 × max(Q(B, ·)) = 0 + 0.9 × 0.7 = **0.63**.

They agree here because "left" is the greedy action in B. They differ when the next chosen action is not greedy (e.g. during ε-greedy exploration).
{{< /collapse >}}

---

**C3.** Compute the 2-step return G_t^{(2)} for rewards [R_{t+1}=0, R_{t+2}=1] and V(S_{t+2})=0.4. Use γ=0.9.

{{< pyrepl code="gamma = 0.9\nR1, R2 = 0, 1\nV_s2 = 0.4\n\nG2 = R1 + gamma * R2 + gamma**2 * V_s2\nprint(f'2-step return = {G2:.4f}')   # 0 + 0.9 + 0.81*0.4 = 1.224" height="160" >}}

{{< collapse summary="Answer" >}}
G^{(2)} = 0 + 0.9 × 1 + 0.81 × 0.4 = 0.9 + 0.324 = **1.224**.
{{< /collapse >}}

---

**C4.** An agent has k=4 actions and uses ε-greedy with ε=0.1. What is the probability of selecting the greedy action?

{{< pyrepl code="k = 4\nepsilon = 0.1\n\n# Probability of greedy = (1 - epsilon) + epsilon/k\n# (greedy because not exploring) + (greedy by luck when exploring)\np_greedy = (1 - epsilon) + epsilon / k\nprint(f'P(greedy) = {p_greedy:.4f}')   # 0.9 + 0.025 = 0.925" height="160" >}}

{{< collapse summary="Answer" >}}
P(greedy) = (1 − ε) + ε/k = 0.9 + 0.1/4 = 0.9 + 0.025 = **0.925**.

Each non-greedy action has probability ε/k = 0.025.
{{< /collapse >}}

---

**C5.** Compute the incremental mean. Start with Q = 0. Observe rewards [1.2, 0.8, 1.0]. Update using the incremental update rule Q ← Q + (1/n)(R − Q). What is the final Q?

{{< pyrepl code="Q = 0.0\nrewards = [1.2, 0.8, 1.0]\n\nfor n, r in enumerate(rewards, start=1):\n    Q = Q + (1/n) * (r - Q)\n    print(f'After update {n}: Q = {Q:.4f}')\n\nprint(f'Final Q = {Q:.4f}')   # should be 1.0 (mean of [1.2, 0.8, 1.0])" height="200" >}}

{{< collapse summary="Answer" >}}
Step 1: Q = 0 + (1/1)(1.2 − 0) = **1.2**.
Step 2: Q = 1.2 + (1/2)(0.8 − 1.2) = 1.2 − 0.2 = **1.0**.
Step 3: Q = 1.0 + (1/3)(1.0 − 1.0) = **1.0**.

Final Q = **1.0** (equals the arithmetic mean of [1.2, 0.8, 1.0]).
{{< /collapse >}}

---

## Code (K) — Implementation

**K1.** Implement the `td0_update(V, s, r, s_next, alpha, gamma)` function.

{{< pyrepl code="def td0_update(V, s, r, s_next, alpha=0.1, gamma=0.9):\n    # TODO: return updated value of V[s]\n    pass\n\nV = {'A': 0.3, 'B': 0.5, 'C': 0.0}\nprint(td0_update(V, 'A', 0, 'B'))   # 0.315\nprint(td0_update(V, 'B', 1, 'C'))   # 0.545" height="200" >}}

{{< collapse summary="Solution" >}}
```python
def td0_update(V, s, r, s_next, alpha=0.1, gamma=0.9):
    td_error = r + gamma * V[s_next] - V[s]
    return V[s] + alpha * td_error
```
V(A): 0.3 + 0.1 × (0 + 0.9×0.5 − 0.3) = 0.315.
V(B): 0.5 + 0.1 × (1 + 0.9×0.0 − 0.5) = 0.545.
{{< /collapse >}}

---

**K2.** Implement epsilon-greedy action selection given a Q-value list (one value per action).

{{< pyrepl code="import random\nrandom.seed(0)\n\ndef epsilon_greedy(Q_values, epsilon=0.1):\n    \"\"\"Return action index. Q_values is a list of floats.\"\"\"\n    # TODO\n    pass\n\nQ = [0.2, 0.8, 0.5, 0.1]\nresults = [epsilon_greedy(Q, epsilon=0.2) for _ in range(1000)]\nprint('Action 1 (greedy) selected', results.count(1), 'times out of 1000')\n# Expected ~850 times (80% greedy + 5% by chance = 85%)" height="220" >}}

{{< collapse summary="Solution" >}}
```python
def epsilon_greedy(Q_values, epsilon=0.1):
    if random.random() < epsilon:
        return random.randrange(len(Q_values))
    return Q_values.index(max(Q_values))
```
{{< /collapse >}}

---

## Debug (D) — Find and fix the bug

**D1.** This SARSA implementation has a bug. Find and fix it.

```python
def sarsa_update(Q, s, a, r, s_next, a_next, alpha=0.1, gamma=0.9):
    # SARSA: on-policy TD control
    td_target = r + gamma * max(Q[s_next].values())  # Bug!
    Q[s][a] += alpha * (td_target - Q[s][a])
    return Q
```

{{< collapse summary="Answer" >}}
The bug: `max(Q[s_next].values())` uses the **greedy** next action — that is the Q-learning target, not SARSA. SARSA must use the **actual next action** `a_next` (the one the policy will take).

**Fix:**
```python
td_target = r + gamma * Q[s_next][a_next]
```
Using `max` makes it Q-learning (off-policy). SARSA is on-policy: it updates using the next action that was actually sampled from the current policy.
{{< /collapse >}}

---

**D2.** Find the off-by-one error in this episode-end handling:

```python
def run_episode(env, Q, alpha=0.1, gamma=0.9, epsilon=0.1):
    s = env.reset()
    done = False
    while not done:
        a = epsilon_greedy(Q[s], epsilon)
        s_next, r, done = env.step(a)
        # TD update — applied even at terminal step
        td_target = r + gamma * max(Q[s_next].values())  # Bug!
        Q[s][a] += alpha * (td_target - Q[s][a])
        s = s_next
```

{{< collapse summary="Answer" >}}
The bug: when `done=True`, `s_next` is a **terminal state** and `Q[s_next]` should be 0 (no future value). But the code bootstraps off `max(Q[s_next].values())` anyway, which is wrong — it uses whatever stale values happen to be in the Q-table for the terminal state.

**Fix:** guard with `if done`:
```python
td_target = r if done else r + gamma * max(Q[s_next].values())
```
Equivalently, initialize terminal state Q-values to 0 and never update them, or use `V(terminal) = 0` explicitly.
{{< /collapse >}}

---

## Challenge (X)

**X1.** Implement Q-learning on a 3×3 gridworld: start at (0,0), goal at (2,2) with reward +1. Each step costs −0.01. Use ε=0.1, α=0.1, γ=0.99. Train for 5000 episodes and print the learned greedy policy (arrows) and total steps per episode (smoothed).

{{< pyrepl code="import random\nrandom.seed(42)\n\n# 3x3 gridworld setup\nACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right\nACTION_SYMBOLS = ['^','v','<','>']\n\ndef step(s, a):\n    r_new = s[0] + ACTIONS[a][0]\n    c_new = s[1] + ACTIONS[a][1]\n    if 0 <= r_new <= 2 and 0 <= c_new <= 2:\n        s_next = (r_new, c_new)\n    else:\n        s_next = s\n    if s_next == (2, 2):\n        return s_next, 1.0, True\n    return s_next, -0.01, False\n\n# TODO: implement Q-learning\n# Q: dict of (state -> list of 4 Q-values)\n# Train 5000 episodes, then print greedy policy\nprint('Implement Q-learning...')" height="260" >}}

{{< collapse summary="Hint" >}}
1. Initialize Q[(r,c)] = [0.0, 0.0, 0.0, 0.0] for all states.
2. Each episode: start at (0,0), loop until done or max 200 steps.
3. Choose action ε-greedy from Q[s]. Take step. Update Q[s][a] using Q-learning target.
4. After training, for each state print `ACTION_SYMBOLS[argmax(Q[s])]`.

Expected policy: arrows pointing toward (2,2) via the shortest path.
{{< /collapse >}}
