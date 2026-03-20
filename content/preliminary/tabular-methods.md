---
title: "Tabular Methods"
description: "Dynamic programming, Monte Carlo vs TD, on-policy vs off-policy, and Q-learning — with explanations and examples."
date: 2026-03-10T00:00:00Z
draft: false
difficulty: 3
tags: ["tabular methods", "dynamic programming", "Monte Carlo", "TD", "Q-learning", "preliminary"]
keywords: ["tabular methods", "dynamic programming", "Monte Carlo vs TD", "on-policy off-policy", "Q-learning"]
weight: 7
roadmap_icon: "chart"
roadmap_color: "indigo"
roadmap_phase_label: "Topic 7 · Tabular Methods"
---

This page covers the tabular methods you need for the preliminary assessment: policy iteration and value iteration, the difference between Monte Carlo and TD, on-policy vs off-policy learning, and the Q-learning update rule. [Back to Preliminary](../).

---

## Why this matters for RL

When the state and action spaces are small enough, we can store one value per state (or state-action) and update them from experience or from the model. Dynamic programming does this when we know the model; Monte Carlo and TD do it from samples. Q-learning is the canonical off-policy TD method and is the basis of many deep RL algorithms (e.g. DQN). You need to know how these methods differ and how to write the Q-learning update.

### Learning objectives

Name two DP methods for solving MDPs; explain how MC and TD differ in when they update and what they use as target; distinguish on-policy vs off-policy and give examples; write the Q-learning update.

---

## Core concepts

- Policy iteration / Value iteration: DP methods that assume we know transition probabilities. Policy iteration alternates policy evaluation (compute \\(V^\pi\\)) and policy improvement. Value iteration updates values using the Bellman optimality equation until convergence.
- Monte Carlo: Uses full returns (sum of rewards until end of episode) to update value estimates. Updates only after an episode ends.
- TD (Temporal Difference): Updates using a bootstrapped target: current reward plus discounted estimate of next state value (e.g. \\(r + \gamma V(s')\\)). Can update every step; doesn’t need the full return.
- On-policy: Learns about the policy used to generate the data (e.g. SARSA). Off-policy: Learns about a target policy while following a different behavior policy (e.g. Q-learning).
- Q-learning: \\(Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\bigr]\\). Uses the max over next-state actions (target policy is greedy) while the behavior policy can be exploratory (e.g. ε-greedy).

**Illustration (MC vs TD):** Monte Carlo updates only at episode end using the full return; TD updates every step using a bootstrapped target. The chart below shows a typical comparison: average reward per episode over 100 episodes for one-step TD vs MC on the same task.

{{< chart type="line" palette="return" title="Average reward per episode (MC vs TD(0))" labels="0, 25, 50, 75, 100" data="5, 25, 45, 62, 75" xLabel="Episode" yLabel="Mean reward" >}}

---

## Worked problems (with explanations)

### 1. Two DP methods (Q16)

Q: Name two dynamic programming methods for solving MDPs when the model (transition probabilities) is known.

{{< collapse summary="Answer and explanation" >}}
Policy Iteration and Value Iteration.

- Policy iteration: (1) Policy evaluation: compute \\(V^\pi\\) for the current policy \\(\pi\\) (e.g. by iterating the Bellman equation). (2) Policy improvement: for each state, set \\(\pi(s) = \arg\max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')]\\). Repeat until the policy doesn’t change.
- Value iteration: Iterate \\(V(s) \leftarrow \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]\\) until convergence, then derive the greedy policy from \\(V\\).

### Explanation

Both assume we know \\(p(s',r|s,a)\\). Policy iteration explicitly represents and improves a policy; value iteration works only with values and derives the policy at the end. In model-free RL we don’t have \\(p\\), so we use MC or TD instead, which learn from experience.
{{< /collapse >}}

---

### 2. Monte Carlo vs TD (Q17)

Q: What is the key difference between Monte Carlo and Temporal Difference (TD) learning in terms of updating value estimates?

{{< collapse summary="Answer and explanation" >}}
Monte Carlo waits until the end of an episode to compute the full return \\(G_t = r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t} r_T\\) and then updates the value estimate using \\(G_t\\) as the target. So the update uses the *actual* return from that trajectory.

TD uses bootstrapping: it updates using the immediate reward plus the *current estimate* of the value of the next state, e.g. target \\(r_t + \gamma V(s_{t+1})\\), without waiting for the episode to finish. So the target is a mix of one real reward and an estimate.

### Explanation

MC is unbiased (given the trajectory) but has high variance and must wait for episode end. TD has lower variance (one step of randomness) but is biased because it uses an estimate (\\(V(s')\\)) instead of the true return. TD can also learn from incomplete episodes and from continuing tasks. In practice, TD methods (including Q-learning) are very common because they update every step and often learn faster.
{{< /collapse >}}

---

### 3. On-policy vs off-policy (Q18)

Q: Explain the difference between on-policy and off-policy learning. Give one algorithm example for each.

{{< collapse summary="Answer and explanation" >}}
- On-policy: The agent learns about the *same* policy it uses to select actions (the behavior policy). The value or Q-function being learned is for the policy that generated the data. Example: SARSA — it updates \\(Q(s,a)\\) using the action actually taken in the next state (from the current policy), so it learns the value of the behavior policy.
- Off-policy: The agent learns about a *target* policy (e.g. greedy) while following a *different* behavior policy (e.g. ε-greedy for exploration). The data is generated by the behavior policy, but the updates target the optimal or another policy. Example: Q-learning — it updates \\(Q(s,a)\\) using \\(\max_{a'} Q(s',a')\\) (greedy next action), so it learns the optimal Q-function even though the behavior policy may be exploratory.

### Explanation

On-policy methods are often simpler but must balance exploration in the same policy we evaluate. Off-policy methods can reuse data (e.g. replay buffers) and learn an optimal policy while exploring, but they can be less stable (e.g. need careful target networks in DQN).
{{< /collapse >}}

---

### 4. Q-learning update rule (Q19)

Q: Write the Q-learning update rule for a transition \\((s, a, r, s')\\).

{{< collapse summary="Answer and explanation" >}}
\\(Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\bigr]\\).

Here \\(\alpha\\) is the step size (learning rate), \\(\gamma\\) is the discount factor, and \\(r + \gamma \max_{a'} Q(s', a')\\) is the TD target. The term in brackets is the TD error: how much the target differs from the current estimate.

### Explanation

We observe \\((s, a, r, s')\\). The target for \\(Q(s,a)\\) is “immediate reward plus discounted value of the best we can do from \\(s'\\),” i.e. \\(r + \gamma \max_{a'} Q(s',a')\\). We move our estimate \\(Q(s,a)\\) a fraction \\(\alpha\\) of the way toward this target. Over many such updates (with sufficient exploration), \\(Q\\) converges to the optimal action-value function \\(Q^*\\).
{{< /collapse >}}

---

## Code example: Q-learning update

```python
def q_learning_update(Q, s, a, r, s_prime, actions, alpha=0.1, gamma=0.99):
    # Q: dict or array (s,a) -> value; actions: list of possible actions
    target = r + gamma * max(Q[s_prime, a_prime] for a_prime in actions)
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
```

### Explanation

We compute the TD target as \\(r + \gamma \max_{a'} Q(s',a')\\); here \\(actions\\) is the set of possible actions so we can take the max over \\(Q(s',a')\\). Then we perform the update: new estimate = old estimate + \\(\alpha\\) × (target − old estimate). This is one step of Q-learning. In practice, \\(Q\\) might be a table (tabular) or a neural network (DQN); the same formula applies, with the network outputting \\(Q(s,a)\\) and the update done via gradient descent on the squared TD error.

---

## Professor's hints

- Q-learning is off-policy because the target uses \\(\max_{a'} Q(s',a')\\) (greedy) while the behavior that produced \\((s,a,r,s')\\) might be ε-greedy or other.
- SARSA uses \\(Q(s', a')\\) where \\(a'\\) is the action actually taken in \\(s'\\); that makes it on-policy.
- In tabular settings, “convergence” of Q-learning to \\(Q^*\\) requires that all state-action pairs are visited infinitely often (e.g. with ε-greedy exploration).

---

## Common pitfalls

- Using the wrong target: Q-learning uses \\(\max_{a'} Q(s',a')\\); SARSA uses \\(Q(s', a')\\) for the action \\(a'\\) that was taken. Mixing them changes the algorithm and what it converges to.
- Forgetting the discount: The target must be \\(r + \gamma \max_{a'} Q(s',a')\\), not \\(r + \max_{a'} Q(s',a')\\).
- Confusing “tabular” with “model-free”: Tabular means we store one value per state (or state-action). We can do tabular MC or tabular TD; both are model-free. DP is typically tabular but requires a model.
