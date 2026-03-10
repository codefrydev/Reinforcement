---
title: "Phase 2 Readiness Quiz"
description: "5–8 questions to check readiness after prerequisites. Solutions included."
date: 2026-03-10T00:00:00Z
draft: false
---

Use this quiz after working through [Python](../prerequisites/python/), [NumPy](../prerequisites/numpy/), and [PyTorch](../prerequisites/pytorch/) (and optionally [Gym](../prerequisites/gym/)). If you can answer at least 6 correctly, you are ready for [Phase 3](../learning-path/#phase-3--rl-foundations) and [Volume 1](../curriculum/volume-01/).

---

### 1. Python

**Q:** What is the output of `[x**2 for x in range(4)]`?

{{< collapse summary="Answer" >}}
`[0, 1, 4, 9]` — list comprehension over 0, 1, 2, 3.
{{< /collapse >}}

---

### 2. Python

**Q:** A trajectory is stored as a list of tuples `(state, action, reward)`. Write one line of Python to extract the list of rewards.

{{< collapse summary="Answer" >}}
`rewards = [t[2] for t in trajectory]` or `rewards = [r for s, a, r in trajectory]`.
{{< /collapse >}}

---

### 3. NumPy

**Q:** Given `arr = np.array([1, 2, 3, 4, 5])`, how do you compute the mean? How do you get the last two elements?

{{< collapse summary="Answer" >}}
Mean: `np.mean(arr)` or `arr.mean()`. Last two: `arr[-2:]` → array([4, 5]).
{{< /collapse >}}

---

### 4. NumPy

**Q:** For `rewards = np.array([0, 0, 1])` and `gamma = 0.9`, write a one-liner (using NumPy, no loop) to compute the discounted return \\(G = r_0 + \\gamma r_1 + \\gamma^2 r_2\\).

{{< collapse summary="Answer" >}}
`G = np.sum((gamma ** np.arange(3)) * rewards)` or `np.dot(gamma ** np.arange(3), rewards)`. Result: 0.81.
{{< /collapse >}}

---

### 5. PyTorch

**Q:** In PyTorch, how do you create a 1D tensor of size 4 that requires gradients? After computing `y = x.sum()` and calling `y.backward()`, what is stored in `x.grad`?

{{< collapse summary="Answer" >}}
`x = torch.tensor([1., 2., 3., 4.], requires_grad=True)`. After `y = x.sum()`, `y.backward()`, we get `x.grad = tensor([1., 1., 1., 1.])` (gradient of sum w.r.t. each element is 1).
{{< /collapse >}}

---

### 6. PyTorch

**Q:** What is the purpose of `optimizer.zero_grad()` before `loss.backward()`?

{{< collapse summary="Answer" >}}
PyTorch *accumulates* gradients by default. `zero_grad()` clears the previous step’s gradients so we don’t add them to the new ones.
{{< /collapse >}}

---

### 7. Gym

**Q:** What does `env.step(action)` return? What do you use to know when the episode is over?

{{< collapse summary="Answer" >}}
It returns `(obs, reward, terminated, truncated, info)`. The episode is over when `terminated or truncated` is True.
{{< /collapse >}}

---

### 8. Gym

**Q:** Write a minimal loop that runs one episode: reset, then step with random actions until done. What variable(s) do you sum to get the episode return?

{{< collapse summary="Answer" >}}
`obs, info = env.reset(); total = 0; done = False; while not done: obs, r, term, trunc, info = env.step(env.action_space.sample()); total += r; done = term or trunc`. Sum the `reward` from each step → episode return.
{{< /collapse >}}

---

**Next step:** If you passed, go to [Phase 3 — RL foundations](../learning-path/#phase-3--rl-foundations) and start [Volume 1](../curriculum/volume-01/).
