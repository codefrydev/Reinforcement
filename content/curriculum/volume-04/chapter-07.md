---
title: "Chapter 37: Asynchronous Advantage Actor-Critic (A3C)"
description: "A3C with multiprocessing workers; compare speed with A2C."
date: 2026-03-10T00:00:00Z
weight: 37
draft: false
tags: ["A3C", "asynchronous", "multiprocessing", "actor-critic", "curriculum"]
keywords: ["A3C", "asynchronous advantage actor-critic", "multiprocessing", "A2C"]
---

**Learning objectives**

- Implement **A3C**: multiple **worker processes** each running an environment and asynchronously updating a **global** shared network.
- Understand the trade-off: A3C can be faster on multi-core CPUs (no synchronization wait) but is often less stable than A2C due to asynchronous gradient updates.
- Compare **training speed** (wall clock and/or sample efficiency) of A3C vs A2C on CartPole.

**Concept and real-world RL**

**A3C** (Asynchronous Advantage Actor-Critic) runs multiple workers in parallel, each collecting experience and pushing gradient updates to a global network. Workers do not wait for each other, so gradients are asynchronous and potentially stale. In **game AI** and early deep RL, A3C was popular for leveraging many CPU cores; in practice, **A2C** (synchronous) or **PPO** often give more stable and reproducible results. The idea of parallel envs and shared parameters remains central; the main difference is sync (A2C) vs async (A3C) updates.

**Where you see this in practice:** A3C appears in classic papers and older codebases; modern implementations often prefer A2C or PPO with vectorized envs (e.g. 8 envs, sync updates).

**Illustration (async vs sync):** A3C workers update a global network asynchronously. The chart below shows a typical reward curve with 4 workers (reward per episode, mixed from all workers).

{{< chart type="line" palette="return" title="Episode return (A3C, 4 workers)" labels="0, 200, 400, 600, 800" data="30, 100, 160, 190, 198" xLabel="Episode" yLabel="Return" >}}

**Exercise:** Implement A3C using Python's multiprocessing. Create several worker processes that each interact with its own environment and update a global network asynchronously. Test on CartPole and compare training speed with A2C.

**Professor's hints**

- Use `torch.multiprocessing` or `multiprocessing`. Global network lives in the main process or in a shared memory structure; workers copy parameters (or use shared tensors), run a few steps, compute gradients, and send gradients (or parameter deltas) back to the main process to apply.
- Async: each worker updates the global network when it finishes its local rollout; no lock-step. Be careful with shared model parameters—use a lock when updating or use a queue to send gradients to the main process.
- CartPole: 4–8 workers is enough to see the effect. Compare total steps to solve (e.g. reach 195 avg return) and wall-clock time for A3C vs A2C with the same number of envs.

**Common pitfalls**

- **Race conditions:** If two workers update the global network simultaneously, parameters can be corrupted. Use a lock or a single update queue.
- **Stale gradients:** In A3C, by the time a worker’s gradient is applied, the global policy may have changed. This can increase variance; A2C avoids this by syncing.

{{< collapse summary="Worked solution (warm-up: A3C vs A2C)" >}}
**Warm-up:** Main advantage of A3C: parallel workers collect experience asynchronously, so we get more diverse data and can scale with more CPUs. Main disadvantage: gradients are computed on stale policy copies (workers may be many steps behind the global network), which can increase variance and hurt stability. A2C waits for all workers to finish a step then updates once with synchronized gradients, trading some parallelism for stability.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence each, what is the main advantage of A3C over A2C, and what is the main disadvantage?
2. **Coding:** Implement A3C with 4 workers. Log the global episode return (from any worker) every 100 updates. Run for 2000 updates and plot; compare with A2C (4 envs) for 2000 updates.
3. **Challenge:** Implement a **synchronous** version that collects rollouts from all workers and then does one batched update (this is A2C). Compare gradient norm and return variance over training for A3C vs this A2C.
