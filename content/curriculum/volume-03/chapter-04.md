---
title: "Chapter 24: Experience Replay"
description: "Replay buffer class with push and sample."
date: 2026-03-10T00:00:00Z
weight: 24
draft: false
tags: ["experience replay", "replay buffer", "DQN", "curriculum"]
keywords: ["experience replay", "replay buffer", "push and sample", "DQN"]
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
