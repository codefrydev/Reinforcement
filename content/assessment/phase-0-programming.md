---
title: "Phase 0 Assessment: Python Basics"
description: "10 Python questions to check readiness after Phase 0. Includes writing functions, finding bugs, and predicting output."
date: 2026-03-19T00:00:00Z
draft: false
tags: ["assessment", "phase 0", "python", "self-check", "solutions"]
keywords: ["phase 0 assessment", "python basics quiz", "RL programming check", "solutions", "10 questions"]
weight: 1
roadmap_icon: "terminal"
roadmap_color: "blue"
roadmap_phase_label: "Phase 0 Quiz"
---

Use this self-check after completing [Phase 0: Programming from Zero](../learning-path/phase-0/). If you can answer at least 8 of 10 correctly, you are ready to move on.

---

### 1. Predict the output

What does this print?

```python
x = 5
if x > 3:
    print("A")
elif x > 1:
    print("B")
else:
    print("C")
```

{{< collapse summary="Answer" >}}
**A** — The first condition `x > 3` is True (5 > 3), so Python executes the first branch and skips the rest.
{{< /collapse >}}

---

### 2. Write a function

Write `clamp(x, lo, hi)` that returns `lo` if x < lo, `hi` if x > hi, otherwise x. (Used in PPO to clip ratios.)

{{< pyrepl code="def clamp(x, lo, hi):\n    pass  # TODO\n\nprint(clamp(1.5, 0.8, 1.2))  # 1.2\nprint(clamp(0.5, 0.8, 1.2))  # 0.8\nprint(clamp(1.0, 0.8, 1.2))  # 1.0" height="220" >}}

{{< collapse summary="Answer" >}}
```python
def clamp(x, lo, hi):
    if x < lo: return lo
    if x > hi: return hi
    return x
```
{{< /collapse >}}

---

### 3. Find the bug

This should print the sum of a rewards list but has a bug. Find it.

```python
rewards = [0.5, 0.3, 0.2]
total = 0
for r in rewards
    total = total + r
print(total)
```

{{< collapse summary="Answer" >}}
Missing colon after `for r in rewards`. Fix: `for r in rewards:` (Python requires `:` after loop/if/def headers). SyntaxError without it.
{{< /collapse >}}

---

### 4. Predict the output

```python
q = [0.1, 0.5, 0.3]
print(q[1])
print(q[-1])
print(len(q))
```

{{< collapse summary="Answer" >}}
`0.5` (index 1), `0.3` (last element, index -1), `3` (length).
{{< /collapse >}}

---

### 5. Write a function

Write `discounted_return(rewards, gamma)` that computes G = r₀ + γr₁ + γ²r₂ + ⋯. Test with rewards=[0,0,1], gamma=0.9 (expected: 0.81).

{{< pyrepl code="def discounted_return(rewards, gamma):\n    pass\n\nprint(discounted_return([0, 0, 1], 0.9))   # 0.81\nprint(discounted_return([1, 0, 0], 0.9))   # 1.0" height="200" >}}

{{< collapse summary="Answer" >}}
```python
def discounted_return(rewards, gamma):
    return sum(gamma**t * r for t, r in enumerate(rewards))
```
{{< /collapse >}}

---

### 6. Find the bug

```python
def epsilon_greedy(Q, epsilon=0.1):
    import random
    if random.random() > epsilon:
        return random.randrange(len(Q))
    return Q.index(max(Q))
```

{{< collapse summary="Answer" >}}
Condition is **reversed**: `> epsilon` means mostly exploring (wrong). Fix: `if random.random() < epsilon:` for exploration, `else: return Q.index(max(Q))` for exploitation.
{{< /collapse >}}

---

### 7. Predict the output

```python
config = {"gamma": 0.99, "epsilon": 0.1}
config["lr"] = 0.001
print(len(config))
print("alpha" in config)
```

{{< collapse summary="Answer" >}}
`3` (three keys: gamma, epsilon, lr), `False` ("alpha" is not a key).
{{< /collapse >}}

---

### 8. Write a function

Write `max_q(Q_dict, state, n_actions)` that returns the maximum Q-value for a given state. Q_dict maps (state, action) → float. Use `Q_dict.get((state, a), 0.0)` for each action.

{{< pyrepl code="def max_q(Q_dict, state, n_actions):\n    pass\n\nQ = {((0,0),1): 0.5, ((0,0),3): 0.7}\nprint(max_q(Q, (0,0), 4))   # 0.7" height="200" >}}

{{< collapse summary="Answer" >}}
```python
def max_q(Q_dict, state, n_actions):
    return max(Q_dict.get((state, a), 0.0) for a in range(n_actions))
```
{{< /collapse >}}

---

### 9. Find the bug

```python
def run_episode(rewards, gamma=0.9):
    G = 0
    for t in range(len(rewards)):
        G = G + gamma * rewards[t]   # bug
    return G
```

{{< collapse summary="Answer" >}}
The discount is wrong: `gamma * rewards[t]` should be `gamma**t * rewards[t]`. As written, every reward is discounted by exactly γ regardless of step. Fix: `G += gamma**t * rewards[t]`.
{{< /collapse >}}

---

### 10. Predict the output

```python
def f(x, y=2):
    return x * y

print(f(3))
print(f(3, 3))
print(f(y=4, x=2))
```

{{< collapse summary="Answer" >}}
`6` (3×2), `9` (3×3), `8` (x=2, y=4 → 2×4). Default argument y=2 is used when not specified.
{{< /collapse >}}

---

**Score:** 8–10: Ready for Phase 1. 6–7: Review the specific topics you missed. Below 6: Complete [Phase 0](../learning-path/phase-0/) and the [Python Confidence Builder](../learning-path/phase-0/python-confidence/) before continuing.
