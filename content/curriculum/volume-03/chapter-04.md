---
title: "Chapter 24: Experience Replay"
description: "Replay buffer class with push and sample."
date: 2026-03-10T00:00:00Z
weight: 24
draft: false
difficulty: 7
tags: ["experience replay", "replay buffer", "DQN", "curriculum"]
keywords: ["experience replay", "replay buffer", "push and sample", "DQN"]
roadmap_color: "green"
roadmap_icon: "chart"
roadmap_phase_label: "Vol 3 · Ch 4"
---

**Learning objectives**

- Implement a **replay buffer** that stores transitions \\((s, a, r, s', \\text{done})\\) with a fixed capacity.
- Use a **circular buffer** (overwrite oldest when full) and **random sampling** for minibatches.
- Test the buffer with random data and verify shapes and sampling behavior.

**Concept and real-world RL**

**Experience replay** stores past transitions and samples random minibatches for training. It breaks the correlation between consecutive samples (which would cause unstable updates if we trained only on the last transition) and reuses data for sample efficiency. DQN and many off-policy algorithms rely on it. The buffer is usually a circular buffer: when full, new transitions overwrite the oldest. Sampling uniformly at random (or with prioritization in advanced variants) gives unbiased minibatches. In practice, buffer size is a hyperparameter (e.g. 10k–1M); too small limits diversity, too large uses more memory and can slow learning if the policy has changed a lot.

**Illustration (buffer fill):** As the agent interacts with the env, the replay buffer fills until it reaches capacity, then the oldest transitions are overwritten. The chart below shows how buffer size grows over the first 15k steps (capacity 10k).

{{< chart type="line" palette="return" title="Replay buffer size over env steps" labels="0, 5k, 10k, 12k, 15k" data="0, 5000, 10000, 10000, 10000" xLabel="Env step" yLabel="Buffer size" >}}

**Exercise:** Code a replay buffer class with methods `push(state, action, reward, next_state, done)` and `sample(batch_size)`. Ensure it uses a circular buffer and random sampling. Test it with random data.

**Professor's hints**

- Storage: use a list of tuples or a NumPy array. For a list, when `len(buffer) >= capacity`, pop the first (or use a deque with maxlen). For an array, keep an index `pos` and do `buffer[pos % capacity] = (s, a, r, s', done)`; increment `pos`.
- Sample: use `random.sample(range(len(buffer)), min(batch_size, len(buffer)))` to get indices, then return a list or batch of (s, a, r, s', done). For batching with PyTorch, stack states into a tensor (batch, state_dim), etc.
- Test: push 100 random transitions (e.g. state = random vector, action = 0 or 1, etc.). Sample 32; check that you get 32 transitions and that states/actions have the expected shapes. Push 1000 more and confirm old ones are dropped (if you track which were added when).

**Common pitfalls**

- **Sampling before buffer has batch_size:** If `len(buffer) < batch_size`, sample only `len(buffer)` or wait until enough. Many implementations sample `min(batch_size, len(buffer))` and the training loop checks `len(buffer) >= batch_size` before updating.
- **Mutable state:** If you store references to state/next_state and the env reuses the same array, you can corrupt the buffer. Copy states when pushing: `buffer.push(s.copy(), a, r, s_next.copy(), done)` (or `torch.clone` for tensors).
- **Order:** Circular buffer does not preserve order of insertion in the storage; that is fine. Sampling is random anyway.

{{< collapse summary="Worked solution (warm-up: replay buffer length)" >}}
**Warm-up:** After pushing 50 transitions with capacity 100, what is `len(buffer)`? After pushing 150 total? **Answer:** After 50: `len(buffer) = 50`. After 150: with a circular buffer of capacity 100, we keep only the most recent 100, so `len(buffer) = 100`. The buffer is full and the oldest 50 have been overwritten. This is why we need a fixed capacity and an index that wraps; sampling is uniform over the current buffer contents.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** After pushing 50 transitions with capacity 100, what is `len(buffer)`? After pushing 150 total, what is `len(buffer)`? (50, then 100 if circular.)
2. **Coding:** Implement a replay buffer class: push(transition), sample(batch_size) returning a batch of (s, a, r, s', done). Use a circular buffer (deque or list with index). Test with 100 pushes and 32 samples.
3. **Challenge:** Add a method `sample_sequential(n)` that returns `n` consecutive transitions (e.g. for n-step learning or RNN). Ensure you do not cross an episode boundary (done=True) if you want full n-step returns.
4. **Variant:** Reduce capacity to 10 and push 50 transitions. What is the minimum and maximum age of a transition in the buffer? How does a very small buffer affect learning diversity?
5. **Debug:** The code below stores a reference to the state array instead of a copy, so all entries end up pointing to the same (most recently mutated) state. Fix it.

{{< pyrepl code="from collections import deque\nimport random\n\nbuffer = deque(maxlen=100)\nstate = [0.0, 0.0]  # mutable list\n\nfor i in range(5):\n    state[0] = float(i)\n    # BUG: stores reference, not copy\n    buffer.append((state, i % 2, float(i), state, False))\n\n# All entries show state = [4.0, 0.0] because of shared reference\nprint('All states same?', all(b[0][0] == 4.0 for b in buffer))\n\n# Fix: store state.copy() or tuple(state)\nbuffer2 = deque(maxlen=100)\nfor i in range(5):\n    state[0] = float(i)\n    buffer2.append((state.copy(), i % 2, float(i), state.copy(), False))\nprint('Fixed states different?', buffer2[0][0][0] != buffer2[-1][0][0])" height="260" >}}

6. **Conceptual:** Why does random sampling from a replay buffer reduce the correlation between training samples? What happens if we train only on the most recent transitions?
7. **Recall:** Explain in one sentence why experience replay improves sample efficiency in off-policy RL.
