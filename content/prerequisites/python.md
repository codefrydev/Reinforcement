---
title: "Python"
description: "Python basics for RL: data structures, classes, functions, control flow, and OOP."
date: 2026-03-10T00:00:00Z
weight: 10
draft: false
tags: ["Python", "data structures", "OOP", "prerequisites"]
keywords: ["Python for RL", "data structures", "classes", "functions", "control flow", "OOP"]
---

Concepts used in the curriculum and in [Preliminary: Python basics](../preliminary/python-basics/): **data structures** (list, tuple, dict, set), **classes and objects**, functions, list comprehensions, loops, and conditionals. RL code is full of trajectories, configs, and custom types (agents, buffers)—all built from these basics.

---

## Data structures

Choosing the right structure makes code clearer and often faster. In RL you’ll use all four constantly.

### List — ordered, mutable

Use for sequences: trajectory of states, batch of indices, rewards per episode.

```python
rewards = []           # empty list
rewards.append(1.0)    # add one element
rewards.extend([2, 3]) # add several → [1.0, 2, 3]
rewards[0]             # first element (index 0)
rewards[-1]            # last element
rewards[1:3]           # slice: indices 1, 2
len(rewards)           # length
# Trajectory: list of (state, action, reward)
traj = [(s0, a0, r0), (s1, a1, r1)]
```

### Tuple — ordered, immutable

Use for fixed-size records: one transition \\((s, a, r, s')\\), coordinates, or multiple return values.

```python
transition = (state, action, reward, next_state)
s, a, r, s_next = transition   # unpacking
coords = (0, 0)                 # (row, col) in gridworld
def get_min_max(x):
    return (min(x), max(x))     # return two values
```

### Dict — key–value, mutable

Use for configs, Q-tables (state/action → value), and any key-based lookup.

```python
config = {"lr": 1e-3, "gamma": 0.99}
config["epsilon"] = 0.1
value = config.get("epsilon", 0.0)   # default if missing
"lr" in config                       # True
for key, val in config.items():
    print(key, val)
# Q-table: (state, action) -> float
Q = {}
Q[((0, 0), 0)] = 0.5
```

### Set — unordered, unique elements, mutable

Use for “unique states visited,” deduplication, or membership tests.

```python
visited = set()
visited.add((0, 0))
visited.add((1, 0))
(0, 0) in visited   # True
len(visited)        # 2
# Unique actions taken in a run
actions_taken = set([0, 1, 0, 1, 2])   # → {0, 1, 2}
```

---

## Classes and objects

RL code often groups data and behavior into **classes**: agents, replay buffers, environments. You need to read and write simple classes.

### Defining a class: `__init__` and `self`

```python
class Agent:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon   # store as instance attribute
        self.steps = 0

    def act(self, state):
        self.steps += 1
        if random.random() < self.epsilon:
            return random.randint(0, 1)   # random action
        return 0   # placeholder: greedy action

agent = Agent(epsilon=0.2)
action = agent.act(some_state)
print(agent.steps)
```

### Storing data and exposing it

```python
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []   # list of (s, a, r, s', done)

    def push(self, s, a, r, s_next, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s_next, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in indices]

buf = ReplayBuffer(100)
buf.push(0, 1, 0.5, 1, False)
len(buf)           # 1 (uses __len__)
batch = buf.sample(1)
```

### Why classes in RL

- **Agent** — Holds policy/network, epsilon, step count; has `.act()`, `.update()`.
- **ReplayBuffer** — Holds list of transitions; has `.push()`, `.sample()`.
- **Config** — You can use a dict, or a class with attributes for clarity and tab-completion.

---

## Worked examples

**Example 1 — Extract rewards from a trajectory.** A trajectory is a list of `(state, action, reward)` tuples. Write one line to get the list of rewards.

{{< collapse summary="Solution" >}}
**Step 1:** Each element is a tuple `(s, a, r)`; the reward is at index 2. **Step 2:** List comprehension: `rewards = [t[2] for t in trajectory]`. Or unpack: `rewards = [r for s, a, r in trajectory]`. **In RL:** We use this to compute returns, plot reward curves, or pass rewards into a training step.
{{< /collapse >}}

**Example 2 — Moving average.** Write a function that returns the moving average of a list with window size 3. For `[1, 2, 3, 4, 5]` the result is `[2.0, 3.0, 4.0]`.

{{< collapse summary="Solution" >}}
**Step 1:** Slide a window of length 3: indices 0–2, 1–3, 2–4. **Step 2:** For each start index `i`, take the mean of `arr[i:i+3]`. **Code:** `def moving_average(arr, window=3): return [sum(arr[i:i+window])/window for i in range(len(arr)-window+1)]`. **In RL:** We use this to smooth learning curves (episode return over time).
{{< /collapse >}}

---

## Why Python matters for RL

- **Functions** — Encapsulate environment step logic, policy evaluation, and training loops. Clean functions make debugging and reuse easier.
- **Lists and list comprehensions** — Store trajectories (states, actions, rewards), batch indices, and rolling statistics. Comprehensions keep code concise.
- **Loops** — `for` over episodes and timesteps; `while` for "until done" or convergence.
- **Dictionaries** — Map state/action keys to values (e.g. simple Q-tables, configs, logs).
- **Conditionals** — Terminal checks, exploration vs. exploitation branches, clipping.

---

## Core concepts with examples

### Functions and default arguments

```python
def moving_average(arr, window=3):
    """Return list of rolling means with given window size."""
    return [sum(arr[i:i+window])/window for i in range(len(arr)-window+1)]

# Usage
moving_average([1, 2, 3, 4, 5])        # [2.0, 3.0, 4.0]
moving_average([1, 2, 3, 4, 5], 2)     # [1.5, 2.5, 3.5, 4.5]
```

### List comprehensions and ranges

```python
# Episode rewards (dummy)
rewards = [0.1 * t + 0.5 for t in range(100)]

# Indices of every 10th step (e.g. for logging)
log_steps = [i for i in range(0, 1000, 10)]

# Build a list of (state, action) from two lists
states = [1, 2, 3]
actions = [0, 1, 0]
pairs = [(s, a) for s, a in zip(states, actions)]  # [(1,0), (2,1), (3,0)]
```

### Loops: episodes and steps

```python
num_episodes = 10
for episode in range(num_episodes):
    total_reward = 0
    done = False
    while not done:
        # ... take action, get reward, update state
        total_reward += reward
        done = True  # when terminal
    print(f"Episode {episode} return: {total_reward}")
```

### Dictionaries for config and simple tabular data

```python
config = {"lr": 1e-3, "gamma": 0.99, "epsilon": 0.1}

# Simple Q-table: (state, action) -> value
Q = {}
Q[(0, 0), 0] = 0.5   # state (0,0), action 0
Q[(0, 0), 1] = -0.2
```

### Conditionals: exploration and clipping

```python
import random

def epsilon_greedy_action(q_values, epsilon=0.1):
    if random.random() < epsilon:
        return random.randint(0, len(q_values) - 1)
    return max(range(len(q_values)), key=lambda a: q_values[a])

# Clipping (e.g. for PPO ratio)
ratio = new_prob / old_prob
clipped = max(0.8, min(1.2, ratio))
```

---

## Exercises

**Exercise 1.** Write a function `discounted_return(rewards, gamma)` that takes a list of rewards \\(r_0, r_1, \\ldots, r_{T-1}\\) and returns \\(G_0 = r_0 + \\gamma r_1 + \\gamma^2 r_2 + \\cdots\\). Use a loop (no NumPy). Test with `rewards = [0, 0, 1]` and `gamma = 0.9`; the result should be \\(0 + 0 + 0.81 = 0.81\\).

**Exercise 2.** Write a function that takes a list of numbers and returns both the minimum and maximum in a single pass (one loop). Return a tuple `(min_val, max_val)`.

**Exercise 3.** Using a list comprehension, build a list of the first 20 square numbers \\(1^2, 2^2, \\ldots, 20^2\\). Then write a one-liner that computes the sum of those squares.

**Exercise 4.** Implement a function `count_occurrences(items)` that returns a dictionary mapping each unique element in `items` to how many times it appears. Example: `count_occurrences([1, 2, 1, 2, 3])` → `{1: 2, 2: 2, 3: 1}`.

**Exercise 5.** Write a function `running_mean(x, window)` that returns a list where the \\(i\\)-th element is the mean of `x[i:i+window]` (like a moving average). Handle the case where `len(x) < window` by returning an empty list. Compare with the `moving_average` example above.

**Exercise 6.** (Data structures) (a) Build a list of 5 tuples, each tuple being \\((s, a, r)\\) with dummy integers (e.g. \\((0, 1, 0)\\), \\((1, 0, 1)\\), …). (b) Use a **set** to collect all unique \\(s\\) values that appear in that list. (c) Use a **dict** to map each \\(s\\) to the list of rewards received at that \\(s\\) (e.g. `{0: [0, 0.5], 1: [1]}`). Do it with a loop and `.get(s, [])` and `.append(r)`.

**Exercise 7.** (Classes) Define a class `EpisodeLogger` with `__init__(self)` that initializes an empty list `self.rewards`. Add a method `add_reward(self, r)` that appends `r` to that list, and a method `total_return(self)` that returns the sum of `self.rewards`. Add `def __len__(self): return len(self.rewards)`. Create an instance, call `add_reward` a few times, then print `total_return()` and `len(logger)`.

**Exercise 8.** Write a function `epsilon_greedy(actions, q_values, epsilon)` that with probability `epsilon` returns a random element from `actions`, and with probability `1 - epsilon` returns the action in `actions` with the highest value in `q_values`. Assume `actions` and `q_values` are same-length lists and `q_values[i]` is the value of `actions[i]`.

**Exercise 9.** Implement a tiny **replay buffer interface**: a class with `__init__(self, max_size)` (store a list of transitions, max length `max_size`), `push(self, s, a, r, s_next)` (append a tuple, drop oldest if over max_size), and `sample(self, n)` (return a list of `n` random transitions, or all if fewer than `n`). Use only lists and `random.sample`; no NumPy.

**Exercise 10.** (Challenge) Write a function `trajectory_return(trajectory, gamma)` where `trajectory` is a list of `(state, action, reward)` tuples. Return the discounted return from step 0: \\(G_0 = r_0 + \\gamma r_1 + \\gamma^2 r_2 + \\cdots\\). Use a loop and indexing; then try a one-liner with `sum` and `enumerate`.

---

## Professor's hints

- Prefer **list comprehensions** over appending in a loop when building trajectory lists or reward sequences; they are clearer and often faster.
- Use **tuples** for transitions \\((s, a, r, s')\\) so they are immutable and can be used as dict keys or put in sets if needed.
- Keep **functions small**: one function, one job. In RL, separate "step environment," "select action," and "update agent" into different functions so you can test and reuse them.
- Use a **dict for config** (e.g. `lr`, `gamma`, `epsilon`) so you can pass one object to your training script and change hyperparameters in one place.
- When writing classes (Agent, ReplayBuffer), put all **state** in `__init__` and `self`; keep methods focused on one operation (e.g. `push`, `sample`).

---

## Common pitfalls

- **Mutable default arguments:** Never use `def f(x, items=[])`; the same list is reused across calls. Use `def f(x, items=None)` and `if items is None: items = []` inside.
- **Modifying a list while iterating:** If you loop `for x in lst` and delete or append to `lst`, you can get wrong results or infinite loops. Iterate over a copy (e.g. `for x in lst[:]`) or build a new list.
- **Integer vs float division:** In Python 3, `3 / 2` is `1.5`. Use `//` for integer division. When computing means for rewards, float is usually what you want.
- **Forgetting `self` in class methods:** The first argument of instance methods must be `self`; use `self.attribute` to read or set instance data.
- **Using `=` instead of `==` in conditionals:** `if x = 5` is a syntax error (assignment); use `if x == 5` for comparison.

---

**Docs:** [python.org/docs](https://docs.python.org/3/).
