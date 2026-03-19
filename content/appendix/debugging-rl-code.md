---
title: "How to Debug RL Code"
description: "Practical guide to finding and fixing common RL bugs. Includes 5 find-the-bug exercises with solutions."
date: 2026-03-19T00:00:00Z
draft: false
tags: ["debugging", "RL code", "common bugs", "appendix", "practical guide"]
keywords: ["debug RL code", "RL bugs", "common mistakes", "reward sign bug", "TD target bug"]
---

RL bugs are uniquely hard to catch because a wrong implementation often still trains — it just learns more slowly or converges to a suboptimal policy. This guide covers the most common bugs, how to detect them, and how to fix them.

---

## The Golden Rule: Test on a Trivial Environment First

Before running DQN on Atari, run it on a 2-state MDP you can solve by hand. If your algorithm can't learn that, it won't learn anything. This is called a **sanity check**.

Sanity check targets by algorithm:
- **Q-learning / SARSA**: converge to correct Q-values on a 3×3 gridworld in <1000 episodes.
- **DQN**: solve CartPole (reach 195 average return) in <100k steps.
- **PPO**: solve CartPole in <50k steps.
- **Linear FA**: converge on a 5-state random walk in <5000 episodes.

---

## Common Bug 1: Wrong Sign on Reward

**Symptoms:** Agent learns to avoid the goal; training reward goes down over time.

**Cause:** Reward is negated: `-1` where you meant `+1`, or the reward is computed as `goal - current` when it should be `current - goal`.

**Example bug:**

```python
if state == GOAL:
    reward = -1   # Bug: should be +1
else:
    reward = 0
```

**Fix:** Print a few episode rewards manually. Is the agent rewarded when you expect it to be?

{{< pyrepl code="# Sanity check: print rewards for a hardcoded good trajectory\ndef get_reward(state, next_state, goal=(4,4)):\n    if next_state == goal:\n        return 1   # Check: is this the right sign?\n    return -1      # Step penalty\n\n# Trace a path to the goal\nfor state in [(0,0), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3)]:\n    r = get_reward(state, (state[0]+1, state[1]) if state[0]<4 else (state[0], state[1]+1))\n    print(f'{state} -> reward {r}')" height="260" >}}

---

## Common Bug 2: Not Using the Done Flag in TD Target

**Symptoms:** Agent underestimates values near terminal states; learning is slow and noisy.

**Cause:** When `done=True`, the TD target should be just `r` (no bootstrap). Including `γ*V(s')` for a terminal state adds phantom future value.

**Example bug:**

```python
# Bug: always bootstraps, even at terminal states
target = r + gamma * V[next_state]

# Fix:
target = r if done else r + gamma * V[next_state]
```

{{< pyrepl code="# Demonstrate the bug\ngamma = 0.99\nV = {(4,4): 0.0}   # terminal state, V should stay 0\n\n# Buggy: bootstraps from terminal\nr = 10   # goal reward\nnext_state = (4,4)\ndone = True\n\ntarget_buggy = r + gamma * V[next_state]   # 10 + 0 = 10 (OK here since V=0)\ntarget_correct = r if done else r + gamma * V[next_state]\n\nprint(f'Buggy target: {target_buggy}')\nprint(f'Correct target: {target_correct}')\n# Now imagine V[(4,4)] erroneously became 5 due to earlier bug:\nV[(4,4)] = 5.0\nprint(f'Buggy with V=5: {r + gamma * V[(4,4)]}')   # 14.95 -- wrong!\nprint(f'Correct with V=5: {r if done else r + gamma * V[(4,4)]}')   # 10 -- correct" height="260" >}}

---

## Common Bug 3: Gamma Applied Incorrectly in Return Computation

**Symptoms:** Returns are too high or too low; value estimates don't match.

**Cause:** Common forms of this bug:
1. Forgetting to apply the discount: `G += r` instead of `G += gamma**t * r`
2. Applying γ once instead of cumulatively in a loop

```python
# Bug version 1: no discount
G = sum(rewards)

# Bug version 2: γ applied once, not per step
G = sum(gamma * r for r in rewards)

# Correct:
G = sum(gamma**t * r for t, r in enumerate(rewards))
```

---

## Common Bug 4: Target Network Updated Every Step

**Symptoms:** DQN doesn't converge, loss oscillates wildly.

**Cause:** The target network should update every N steps (e.g. N=100 or N=1000), not every step. Updating every step makes the target non-stationary (the target changes as fast as the main network), removing the stabilization benefit.

**Fix:**

```python
# Bug:
for step in range(total_steps):
    loss = compute_td_loss(...)
    optimizer.step()
    target_net.load_state_dict(policy_net.state_dict())   # Bug: every step!

# Fix:
for step in range(total_steps):
    loss = compute_td_loss(...)
    optimizer.step()
    if step % TARGET_UPDATE_FREQ == 0:   # Only every N steps
        target_net.load_state_dict(policy_net.state_dict())
```

---

## Common Bug 5: Wrong Q-learning Target (Using Current Action)

**Symptoms:** Off-policy Q-learning accidentally becomes on-policy (SARSA).

**Cause:** Using the next action actually taken in the target, instead of the max.

```python
# Bug: SARSA target (on-policy)
next_action = epsilon_greedy(Q[next_state])
target = r + gamma * Q[next_state][next_action]

# Correct Q-learning target (off-policy):
target = r + gamma * max(Q[next_state])
```

---

## Logging Strategy

Add these logs to every RL training loop:

```python
# Every episode
print(f"Episode {ep}: return={total_return:.2f}, steps={steps}, epsilon={epsilon:.3f}")

# Every N steps (for DQN/continuous)
if step % 1000 == 0:
    print(f"Step {step}: loss={loss:.4f}, mean_Q={mean_q:.3f}")

# Periodically
if ep % 100 == 0:
    eval_return = evaluate_greedy(env, Q, n_episodes=10)
    print(f"Eval return: {eval_return:.2f}")
```

**What to watch:**
- Return should generally increase over time (not immediately, but trend upward).
- Loss should decrease or stabilize (not grow indefinitely).
- Q-values should be in a reasonable range (not exploding to ±∞).
- Epsilon should decrease if you're using epsilon decay.

---

## 5 Find-the-Bug Exercises

### Bug Exercise 1

```python
def td_update(V, state, next_state, reward, alpha=0.1, gamma=0.9):
    delta = reward + gamma * V[next_state] - V[state]
    V[next_state] += alpha * delta   # Bug!
    return V
```

{{< collapse summary="Answer" >}}
The update should be applied to **V[state]**, not V[next_state]. Fix: `V[state] += alpha * delta`.
{{< /collapse >}}

---

### Bug Exercise 2

```python
def q_learning_update(Q, s, a, r, s_next, done, alpha=0.1, gamma=0.9):
    if done:
        target = r
    else:
        next_action = max(range(len(Q[s_next])), key=lambda a: Q[s_next][a])
        target = r + gamma * Q[s_next][next_action]   # Bug: this is Q-learning? SARSA?
    Q[s][a] += alpha * (target - Q[s][a])
```

{{< collapse summary="Answer" >}}
This is actually **correct Q-learning** — `next_action = argmax Q[s_next]`, which is the greedy action, not the behaviorally chosen next action. The confusion is in the variable name `next_action`; the code correctly uses max Q-value. However, a common mistaken version would be:

```python
# Bug: pass in a' (behaviorally chosen action) instead of argmax
next_action = actual_next_action_taken   # This would make it SARSA
target = r + gamma * Q[s_next][next_action]
```

To make it unambiguously Q-learning: `target = r + gamma * max(Q[s_next])`.
{{< /collapse >}}

---

### Bug Exercise 3

```python
rewards = [0, 1, 0, 0, 1]
gamma = 0.9
G = 0
for r in reversed(rewards):
    G = r + gamma * G   # Is this correct?
print(G)
```

{{< collapse summary="Answer" >}}
This is **correct** — iterating in reverse and accumulating `G = r + γG` is equivalent to `G_0 = Σ γ^t r_t`. The output is `r4 + γr3 + γ²r2 + ... = 0.9^4 * 0 + ... + 1 + γ*(0 + γ*(0 + γ*(1 + γ*0))) = 1 + 0.9*(1 + 0) = 1.9`. Wait, let me recalculate: rewards=[0,1,0,0,1], reversed=[1,0,0,1,0]. G=0 → G=1+0=1 → G=0+0.9=0.9 → G=0+0.81=0.81 → G=1+0.729=1.729 → G=0+1.556=1.556. This is the correct discounted return from step 0.

**This is not a bug** — it's the correct backward accumulation. Many students think iterating in reverse is wrong; it's actually the standard efficient implementation.
{{< /collapse >}}

---

### Bug Exercise 4

```python
def epsilon_greedy(Q_s, epsilon=0.1, n_actions=4):
    import random
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return max(range(n_actions), key=lambda a: Q_s.get(a, 0))

# Usage:
Q = {0: 0.1, 1: 0.5, 2: 0.3}
# Bug: n_actions not passed, defaults to 4
# But Q only has 3 entries. When action=3 is selected, Q_s.get(3,0) = 0.
# Is this a bug?
```

{{< collapse summary="Answer" >}}
This is a **subtle bug**. If n_actions=4 but Q only has keys 0, 1, 2, action 3 always gets Q-value 0. During epsilon-greedy, action 3 can be chosen (randomly). During greedy, it would only be chosen if all other Q-values are ≤ 0. This causes an **asymmetry**: unexplored actions have value 0 (not the same as "unknown"). Fix: either initialise Q for all actions explicitly, or use consistent n_actions.
{{< /collapse >}}

---

### Bug Exercise 5

```python
# DQN training loop
for step in range(total_steps):
    state = env.current_state()
    action = policy(state, epsilon)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    if len(replay_buffer) > batch_size:
        batch = replay_buffer.sample(batch_size)
        s, a, r, s_next, d = batch
        
        # Compute targets
        targets = r + gamma * target_net(s_next).max(1)[0]   # Bug!
        
        # ... rest of update
```

{{< collapse summary="Answer" >}}
The bug: **done flags are not used in target computation**. When `d=True` (episode ended), the target should be just `r`, not `r + γ * Q_target(s_next)`. Fix:

```python
targets = r + gamma * target_net(s_next).max(1)[0] * (1 - d.float())
# The (1-d) term zeroes out the bootstrap for terminal transitions
```

This is one of the most common DQN bugs and causes instability near episode boundaries.
{{< /collapse >}}
