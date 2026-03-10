---
title: "Phase 2 Readiness Quiz"
description: "5–8 questions to check readiness after prerequisites. Solutions included."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["assessment", "phase 2", "readiness", "prerequisites", "solutions"]
keywords: ["phase 2 readiness", "prerequisites quiz", "Python NumPy PyTorch", "solutions", "readiness check"]
---

Use this quiz after working through [Python](../prerequisites/python/), [NumPy](../prerequisites/numpy/), and [PyTorch](../prerequisites/pytorch/) (and optionally [Gym](../prerequisites/gym/)). If you can answer at least 6 correctly, you are ready for [Phase 3](../learning-path/#phase-3--rl-foundations) and [Volume 1](../curriculum/volume-01/).

---

### 1. Python

**Q:** What is the output of `[x**2 for x in range(4)]`?

{{< collapse summary="Answer" >}}
**Step 1:** `range(4)` gives 0, 1, 2, 3. **Step 2:** `x**2` for each gives 0, 1, 4, 9. **Answer:** `[0, 1, 4, 9]`. List comprehensions are used throughout the curriculum for building lists from trajectories (e.g. rewards, returns).
{{< /collapse >}}

---

### 2. Python

**Q:** A trajectory is stored as a list of tuples `(state, action, reward)`. Write one line of Python to extract the list of rewards.

{{< collapse summary="Answer" >}}
**Step 1:** Each tuple is `(state, action, reward)`, so the reward is index 2. **Step 2:** `[t[2] for t in trajectory]` or unpack: `[r for s, a, r in trajectory]`. **Answer:** `rewards = [t[2] for t in trajectory]`. In RL we often need to extract rewards to compute returns or plot learning curves.
{{< /collapse >}}

---

### 3. NumPy

**Q:** Given `arr = np.array([1, 2, 3, 4, 5])`, how do you compute the mean? How do you get the last two elements?

{{< collapse summary="Answer" >}}
**Mean:** `np.mean(arr)` or `arr.mean()` — both return 3.0. **Last two elements:** `arr[-2:]` uses slicing (from second-to-last to end) → `array([4, 5])`. NumPy arrays are used for states and batches in the curriculum; slicing is used for minibatches and sequences.
{{< /collapse >}}

---

### 4. NumPy

**Q:** For `rewards = np.array([0, 0, 1])` and `gamma = 0.9`, write a one-liner (using NumPy, no loop) to compute the discounted return \\(G = r_0 + \\gamma r_1 + \\gamma^2 r_2\\).

{{< collapse summary="Answer" >}}
**Step 1:** Discount factors: `gamma ** np.arange(3)` = [1, 0.9, 0.81]. **Step 2:** Element-wise product with rewards: [0, 0, 1] * [1, 0.9, 0.81] = [0, 0, 0.81]. **Step 3:** Sum = 0.81. **Code:** `G = np.sum((gamma ** np.arange(len(rewards))) * rewards)`. This is the discounted return \\(G_0\\) used everywhere in RL.
{{< /collapse >}}

---

### 5. PyTorch

**Q:** In PyTorch, how do you create a 1D tensor of size 4 that requires gradients? After computing `y = x.sum()` and calling `y.backward()`, what is stored in `x.grad`?

{{< collapse summary="Answer" >}}
**Create tensor:** `x = torch.tensor([1., 2., 3., 4.], requires_grad=True)` (floats required for grad). **Forward:** `y = x.sum()` ⇒ y = 10. **Backward:** `y.backward()` computes \\(\\partial y/\\partial x_i = 1\\) for each \\(i\\). **Result:** `x.grad = tensor([1., 1., 1., 1.])`. PyTorch accumulates gradients; we call `zero_grad()` before the next backward so they don’t add up.
{{< /collapse >}}

---

### 6. PyTorch

**Q:** What is the purpose of `optimizer.zero_grad()` before `loss.backward()`?

{{< collapse summary="Answer" >}}
PyTorch *accumulates* gradients by default: each `backward()` adds to the existing `.grad` attributes. **Why zero_grad:** We want the gradient for the *current* batch only. So we call `optimizer.zero_grad()` before `loss.backward()` to clear the previous step’s gradients. Otherwise we would be doing gradient descent on a sum of many batches, which is wrong for SGD.
{{< /collapse >}}

---

### 7. Gym

**Q:** What does `env.step(action)` return? What do you use to know when the episode is over?

{{< collapse summary="Answer" >}}
**Return value:** `(obs, reward, terminated, truncated, info)`. **Episode end:** `done = terminated or truncated`. `terminated` = True when the task ends naturally (e.g. goal reached, failure). `truncated` = True when we hit a time limit or other cutoff. In RL we use `done` to mask bootstrap (e.g. no \\(\\gamma V(s')\\) when done) and to reset the env.
{{< /collapse >}}

---

### 8. Gym

**Q:** Write a minimal loop that runs one episode: reset, then step with random actions until done. What variable(s) do you sum to get the episode return?

{{< collapse summary="Answer" >}}
**Step 1:** `obs, info = env.reset(); total = 0; done = False`. **Step 2:** `while not done:` take `action = env.action_space.sample()`, then `obs, r, term, trunc, info = env.step(action)`; add `total += r`; set `done = term or trunc`. **Episode return:** the sum of `reward` from each step (the variable we accumulated in `total`). This is the same structure used for any policy (random or trained) in the curriculum.
{{< /collapse >}}

---

**Next step:** If you passed, go to [Phase 3 — RL foundations](../learning-path/#phase-3--rl-foundations) and start [Volume 1](../curriculum/volume-01/).
