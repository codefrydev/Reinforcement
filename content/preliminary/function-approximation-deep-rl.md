---
title: "Function Approximation and Deep RL"
description: "Why FA, policy gradient update, DQN exploration, experience replay, and actor-critic — with explanations."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["function approximation", "policy gradient", "DQN", "actor-critic", "preliminary"]
keywords: ["function approximation", "policy gradient", "DQN", "experience replay", "actor-critic", "deep RL"]
---

This page covers function approximation and deep RL concepts you need for the preliminary assessment: why we need FA, the policy gradient update, exploration in DQN, experience replay, and the advantage of actor-critic. [Back to Preliminary](../).

---

## Why this matters for RL

In large or continuous state spaces we cannot store a value per state; we use a parameterized function (e.g. neural network) to approximate values or policies. That leads to policy gradient methods (maximize return) and value-based methods with FA (e.g. DQN). DQN uses experience replay and exploration (e.g. ε-greedy); actor-critic combines a policy (actor) and a value function (critic) for lower-variance policy gradients. You need to understand why FA is necessary and how these pieces fit together.

### Learning objectives

Explain why function approximation is needed; write the policy gradient parameter update; name exploration strategies in DQN; explain experience replay and actor-critic.

---

## Core concepts

- Function approximation (FA): Represent \\(V(s)\\) or \\(Q(s,a)\\) (or the policy) with a parameterized function (e.g. \\(V(s; w)\\), \\(Q(s,a; \theta)\\)). We generalize from seen states to unseen ones and can handle huge or continuous spaces.
- Policy gradient: Maximize expected return \\(J(\theta)\\) by gradient ascent: \\(\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)\\). The gradient is given by the policy gradient theorem (involving \\(\nabla_\theta \log \pi(a|s;\theta)\\) and returns or advantages).
- Exploration in DQN: ε-greedy (with probability ε take a random action) and noisy networks (learnable noise in weights) are common.
- Experience replay: Store transitions \\((s,a,r,s')\\) in a buffer and sample random minibatches to train the Q-network. Breaks correlation between consecutive updates and reuses data.
- Actor-critic: The actor is the policy; the critic is a value function (e.g. \\(V(s)\\) or \\(A(s,a)\\)). The critic reduces the variance of policy gradient estimates (e.g. by using a baseline or advantage), leading to faster and more stable learning than plain REINFORCE.

---

## Worked problems (with explanations)

### 1. Why function approximation (Q20)

Q: Why is function approximation necessary in RL for large or continuous state spaces?

{{< collapse summary="Answer and explanation" >}}
Tabular methods store one number per state (or per state-action pair). When the state space is huge (e.g. \\(10^{20}\\) states) or continuous (e.g. \\(\mathbb{R}^n\\)), we cannot store or visit every state. Function approximation uses a parameterized function (e.g. neural network with a fixed number of parameters) to approximate \\(V(s)\\) or \\(Q(s,a)\\) for *any* \\(s\\) (and \\(a\\)). So we generalize from the states we have seen to unseen states; the number of parameters is much smaller than the number of states. That makes learning feasible in large or continuous spaces.

### Explanation

In deep RL, the “function” is usually a neural network. We don’t learn a separate value for each state; we learn weights that map state (and possibly action) to a value. That’s why we can apply RL to images, high-dimensional sensors, and continuous control.
{{< /collapse >}}

---

### 2. Policy gradient update (Q21)

Q: In supervised learning, you minimize a loss function \\(L(\theta)\\) using gradient descent: \\(\theta \leftarrow \theta - \alpha \nabla_\theta L\\). What is the analogous update in policy gradient methods?

{{< collapse summary="Answer and explanation" >}}
In policy gradient we maximize the expected return \\(J(\theta)\\), so we use gradient ascent:
\\(\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)\\).

Here \\(J(\theta)\\) is the expected return (e.g. average reward per episode or expected discounted return), and \\(\nabla_\theta J\\) is given by the policy gradient theorem (it involves \\(\nabla_\theta \log \pi(a|s;\theta)\\) and the return or advantage). So we *add* a multiple of the gradient instead of subtracting, because we want to increase return, not decrease a loss.

### Explanation

In supervised learning we minimize loss (e.g. cross-entropy); in policy gradient we maximize performance. So the sign is opposite: plus for policy gradient, minus for loss minimization. The gradient \\(\nabla_\theta J\\) tells us how to change \\(\theta\\) to get more return; we take a step in that direction.
{{< /collapse >}}

---

### 3. Exploration in Deep RL (Q22)

Q: Name two common exploration strategies used in Deep Q-Networks.

{{< collapse summary="Answer and explanation" >}}
- ε-greedy: With probability \\(\varepsilon\\) take a random action; with probability \\(1-\varepsilon\\) take the action that maximizes \\(Q(s,a)\\). So we usually exploit the current Q-estimate but sometimes explore randomly. \\(\varepsilon\\) is often decayed over time or held fixed (e.g. 0.1).
- Noisy networks: Add learnable noise to the weights (or activations) of the Q-network. The noise is sampled each time we act, so different actions get tried without a separate random action. The noise parameters can be learned (e.g. NoisyNet).

### Explanation

Exploration is needed so we don’t get stuck with a suboptimal policy. ε-greedy is simple and widely used; noisy networks provide state-dependent exploration and can be more sample-efficient. Other options include UCB-style bonuses, intrinsic motivation, and entropy regularization in actor-critic.
{{< /collapse >}}

---

### 4. Experience replay (Q23)

Q: Why is experience replay used in DQN? What problem does it solve?

{{< collapse summary="Answer and explanation" >}}
Experience replay stores many transitions \\((s, a, r, s')\\) in a buffer and samples random minibatches from this buffer to train the Q-network. It addresses two issues:

1. Correlation: Consecutive transitions from the same episode are highly correlated. Updating only on the last transition would make learning unstable (similar states, similar gradients). Sampling *randomly* from the buffer breaks this correlation and makes updates more like i.i.d. training.
2. Sample efficiency: Each transition can be used multiple times for updates, so we get more learning from the same amount of experience.

### Explanation

Without replay, DQN would update on a stream of correlated data and could diverge or learn slowly. Replay makes the training distribution more stationary and diverse. The trade-off is that we learn from off-policy data (old transitions from past policies), which Q-learning already supports because it is off-policy.
{{< /collapse >}}

---

### 5. Actor-critic advantage (Q24)

Q: What is the advantage of using an actor-critic method over pure policy gradient (REINFORCE)?

{{< collapse summary="Answer and explanation" >}}
REINFORCE (pure policy gradient) uses the full return \\(G_t\\) from the current time step as the scale for \\(\nabla_\theta \log \pi(a_t|s_t;\theta)\\). The variance of \\(G_t\\) can be very high (returns vary a lot across trajectories), so learning is slow and unstable.

Actor-critic methods use a critic (a value function \\(V(s)\\) or advantage \\(A(s,a)\\)) to replace or reduce the return in the gradient estimate. For example, we might use \\(A_t = G_t - V(s_t)\\) (advantage = return minus baseline) or a TD-based advantage. The critic reduces variance because it subtracts a baseline (so we only reinforce *better than average*), and/or because it uses lower-variance estimates (e.g. TD) instead of full returns. That typically leads to faster and more stable learning than REINFORCE.

### Explanation

The actor is the policy we improve; the critic tells us “how good was that action?” Using the critic, we don’t need to wait for the full return and we reduce variance, so we can update every step and learn more efficiently. Methods like A2C, A3C, PPO, and SAC are actor-critic style.
{{< /collapse >}}

---

## Code snippet: ε-greedy action (with explanation)

```python
import numpy as np
def epsilon_greedy(Q_s, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)  # random action
    return np.argmax(Q_s)  # greedy action
```

### Explanation

`Q_s` is the vector of Q-values for the current state (one per action). With probability `epsilon` we ignore Q and pick a random action; otherwise we pick the action with the largest Q-value. This is the standard exploration strategy for DQN and many tabular methods. The same idea applies when Q comes from a neural network: we pass the state through the network to get `Q_s`, then apply ε-greedy.
---

## Professor's hints

- Policy gradient: we *maximize* \\(J\\), so update is \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\). Don’t confuse with loss minimization.
- Experience replay is a key ingredient of DQN; target networks (separate network for the TD target) are another; both improve stability.
- Actor-critic = policy (actor) + value function (critic). The critic is used to form a baseline or advantage so the actor’s gradient has lower variance.

---

## Common pitfalls

- Wrong sign for policy gradient: We add the gradient (ascend), not subtract. Subtracting would minimize return.
- Thinking replay is on-policy: Replay uses old data from past policies; Q-learning is off-policy so that’s fine. For on-policy methods (e.g. many actor-critic variants), we usually don’t use replay or use it carefully (e.g. short buffers).
- Confusing “actor” with “behavior policy”: The actor is the policy we are improving. In actor-critic we typically use the same policy to collect data (on-policy) or combine with off-policy corrections.
