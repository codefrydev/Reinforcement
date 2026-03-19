---
title: "Phase 0: Programming from Zero"
description: "Learn programming from scratch: Python installation, variables, conditionals, loops, and functions. No prior experience required."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["phase 0", "programming", "Python", "beginner", "learning path"]
keywords: ["programming from zero", "Python installation", "variables conditionals loops", "beginner programming", "no prior experience"]
---

This phase is for anyone who has **never programmed before**. You will install Python, run your first script, and learn the core ideas (variables, conditionals, loops, functions) that every RL codebase uses. Work through the sections in order.

When you are done, you will be ready for the full [Python prerequisite](../../prerequisites/python/).

---

## What is programming? Why Python for RL?

**Programming** means giving a computer step-by-step instructions. You write **code** in a **programming language**; a program that understands that language runs your code and does what you asked.

**Python** is a language that reads almost like English and is widely used in science and machine learning. In reinforcement learning, researchers and engineers use Python to define environments (e.g. games, simulators), implement agents (policies, neural networks), and run training loops. Learning Python first means you can read and write RL code later.

**In RL we use this when:** Every exercise in this curriculum is implemented in Python. You will write loops that run thousands of episodes and functions that compute rewards and updates.

---

### Practice

1. In one sentence, what does a “program” do?
2. Name one reason Python is used for reinforcement learning.

---

### Professor's hint

Do not try to memorize everything. Focus on understanding one idea at a time and run the code yourself. Typing and running code fixes ideas in your head better than only reading.

---

## Installing Python and running "Hello, World"

You need **Python 3** (3.8 or newer is fine) on your computer.

- **Windows:** Download the installer from [python.org](https://www.python.org/downloads/). During setup, check “Add Python to PATH.”
- **macOS:** Install via [python.org](https://www.python.org/downloads/) or run `brew install python3` if you use Homebrew.
- **Linux:** Use your package manager, e.g. `sudo apt install python3 python3-pip` (Ubuntu/Debian).

Open a **terminal** (Command Prompt on Windows, Terminal on macOS/Linux) and type:

```bash
python3 --version
```

You should see something like `Python 3.10.0`. Now create a file named `hello.py` with this single line:

```python
print("Hello, World")
```

Run it from the terminal (in the folder where `hello.py` is):

```bash
python3 hello.py
```

You should see `Hello, World` printed. You have just run your first program.

**In RL we use this when:** You will run scripts like `python3 train_agent.py` to start training. The same idea: write code in a `.py` file, run it with Python.

---

### Practice

1. Change the message inside `print(...)` to your name and run the script again.
2. Add a second line `print("I am learning Python for RL.")` and run the script. What happens?

---

### Common pitfall

On Windows, if `python3` is not found, try `python` instead. If Python was not added to PATH during installation, you may need to reinstall and check “Add Python to PATH” or add the installation folder to PATH manually.

---

## Variables and types

A **variable** is a name that holds a value. You assign with `=`:

```python
reward = 1.0
step_count = 10
agent_name = "DQN"
```

**Types** describe the kind of value:

- **int** — whole numbers: `0`, `42`, `-3`
- **float** — decimals: `0.99`, `3.14`, `-0.5`
- **str** — text in quotes: `"up"`, `"CartPole"`
- **bool** — True or False: `done = True`, `exploring = False`

You can use variables in expressions and in `print`:

```python
gamma = 0.9
discount = gamma ** 2   # 0.81
print("Discount for 2 steps:", discount)
```

**In RL we use this when:** Rewards, discount factors (\\(\gamma\\)), step counts, and flags like “done” are all stored in variables. States and actions are often numbers or small collections of numbers.

---

### Practice

1. Create variables for your age (int), your height in metres (float), and your name (str). Print them in one sentence.
2. Set `r1, r2, r3 = 0.0, 0.0, 1.0` (three rewards). Write an expression that computes the sum and assign it to `total`, then print `total`.

---

### Professor's hint

Use meaningful names: `total_reward` is better than `x`. In RL code you will see names like `gamma`, `epsilon`, `state`, `action`—they make the code readable.

---

### Common pitfall

Division in Python 3 always gives a float: `4 / 2` is `2.0`. If you need an integer, use `4 // 2` (integer division) or `int(4 / 2)`.

---

## Conditionals (if / else)

**Conditionals** let the computer choose what to do based on whether something is true or false:

```python
reward = 1.0
if reward > 0:
    print("Good outcome")
else:
    print("Bad or neutral outcome")
```

Use `elif` for more cases:

```python
if reward > 0:
    print("Positive")
elif reward < 0:
    print("Negative")
else:
    print("Zero")
```

Indentation (spaces at the start of a line) defines which lines belong to the `if` or `else`. Use 4 spaces consistently.

**In RL we use this when:** Deciding “explore or exploit” (e.g. if random number &lt; ε, take a random action, else take the best action), checking if an episode is done, and clipping gradients or ratios in advanced algorithms.

---

### Practice

1. Write code that sets `score = 85` and prints "Pass" if `score >= 60`, otherwise "Fail".
2. Write code that sets `done = True` and prints "Episode finished" if `done` is True, otherwise "Continue".

---

### Common pitfall

Using `=` (assignment) instead of `==` (comparison). `if x = 5` is wrong and will cause an error; use `if x == 5`.

---

## Loops (for and while)

**Loops** repeat a block of code.

**for** — repeat over a sequence (e.g. a range of numbers):

```python
for step in range(5):
    print("Step", step)   # 0, 1, 2, 3, 4
```

**while** — repeat until a condition is false:

```python
step = 0
while step < 3:
    print("Step", step)
    step = step + 1
```

**In RL we use this when:** The outer loop is often “for each episode,” and the inner loop is “while not done: take action, get reward, update state.” Almost every RL script has these two levels of loops.

---

### Practice

1. Use a `for` loop to print the numbers 1, 2, 3, 4, 5 (hint: `range(1, 6)`).
2. Use a loop to compute the sum of rewards `[0, 0, 1]` and print the sum. Do the same for a list `[0.5, 0.5, 0.5]`.

---

### Professor's hint

`range(n)` gives 0 up to n-1, not 1 to n. So `range(10)` is 0,1,…,9. This is standard in programming and matches “zero-based” indexing (the first element is at index 0).

---

### Common pitfall

**Off-by-one errors:** Check whether your loop should run exactly `n` times (often `range(n)`) or from 1 to n (e.g. `range(1, n+1)`). In RL, “step 0” is the first step, which confuses some beginners.

---

## Functions (defining, calling, return values)

A **function** is a reusable block of code with a name. You **define** it with `def`, then **call** it by name:

```python
def greet(name):
    return "Hello, " + name

message = greet("Alice")
print(message)   # Hello, Alice
```

Functions can take multiple arguments and return one value (or none, or use a tuple to return several):

```python
def add_rewards(r1, r2):
    return r1 + r2

total = add_rewards(0.5, 0.3)   # 0.8
```

**In RL we use this when:** You will write functions for “take one step in the environment,” “choose an action,” “compute discounted return,” and “update the agent.” Breaking code into functions keeps things clear and testable.

---

### Practice

1. Write a function `double(x)` that returns `2 * x`. Call it with `double(5)` and print the result.
2. Write a function `is_positive(r)` that takes a number `r` and returns `True` if `r > 0`, otherwise `False`. Test it with `is_positive(1)` and `is_positive(-1)`.
3. Write a function `sum_list(numbers)` that takes a list of numbers and returns their sum. Test with `sum_list([1, 2, 3])` (should be 6).
4. Write a function `max_q(Q_values)` that takes a list of Q-values and returns the maximum. Test with `[0.1, 0.5, -0.2, 0.3]` (expected: 0.5).
5. Write a function `greedy_policy(Q_values)` that takes a list of Q-values (one per action) and returns the index of the highest value (use `Q_values.index(max(Q_values))`). Test with `[0.1, 0.5, -0.2, 0.3]` (expected: 1).
6. Write a function `run_episode(n_steps, rewards)` that takes a list of rewards and a discount factor `gamma=0.9` and returns the discounted return. Test with `rewards=[0, 0, 1]`, `gamma=0.9` (expected: 0.81).

---

### Professor's hint

Keep functions small and focused. One function, one job. In RL, a function that “steps the environment” should not also be computing the agent’s next action—separate concerns.

---

### Common pitfall

**Mutable default arguments:** Do not use a list as a default value, e.g. `def f(x, items=[])`. The same list is reused across calls. Use `def f(x, items=None)` and then `if items is None: items = []` inside the function.

---

## Lists

A **list** stores an ordered sequence of values. You use lists constantly in RL: rewards per step, states in a trajectory, Q-values for each action.

```python
rewards = []             # empty list
rewards.append(0.0)      # add one reward
rewards.append(1.0)
print(rewards)           # [0.0, 1.0]
print(rewards[0])        # first item (index 0)
print(rewards[-1])       # last item
print(len(rewards))      # number of items
```

You can also **create** a list directly:

```python
states = [(0,0), (0,1), (1,1), (2,2)]   # trajectory of (row, col)
actions = [3, 1, 1]                       # sequence of actions
```

**In RL we use this when:** A trajectory is a list of (state, action, reward) triples. A replay buffer is a list of transitions. Q-values for 4 actions fit in a list of length 4.

---

### Practice

1. Create a list `q_values = [0.2, 0.5, -0.1, 0.3]` (one value per action). Print the highest Q-value using `max(q_values)` and its index using `q_values.index(max(q_values))`.
2. Start with an empty list `episode = []`. Use a loop to append tuples `(step, step * 0.1)` for steps 0, 1, 2, 3, 4. Print the final list.
3. Given `rewards = [0, 0, 1, 0, 1]`, use a loop to count how many rewards are greater than 0.

{{< pyrepl code="q_values = [0.2, 0.5, -0.1, 0.3]\n# TODO: print max Q-value and its index\nprint(max(q_values))" height="200" >}}

---

### Common pitfall

**Index out of range:** If a list has 4 items (indices 0–3), accessing `lst[4]` raises an `IndexError`. Use `len(lst)` to check the size before indexing.

---

## Dictionaries

A **dictionary** maps keys to values. In RL, a Q-table is a dict mapping `(state, action)` pairs to Q-values.

```python
config = {"gamma": 0.99, "epsilon": 0.1, "lr": 0.001}
print(config["gamma"])          # 0.99
config["alpha"] = 0.1           # add new key
print("lr" in config)           # True
```

Iterating over a dict:

```python
for key, value in config.items():
    print(key, "->", value)
```

A Q-table:

```python
Q = {}
Q[((0, 0), 0)] = 0.5   # state=(0,0), action=0, value=0.5
Q[((0, 0), 1)] = -0.2
best_action = max([0, 1], key=lambda a: Q.get(((0,0), a), 0.0))
print("Best action:", best_action)   # 0
```

**In RL we use this when:** Tabular Q-learning stores Q-values in a dict. Agent configurations (learning rate, gamma, epsilon) are often stored as dicts for easy access and logging.

---

### Practice

1. Create a dict `arm_counts = {}` with keys `"arm_0"`, `"arm_1"`, `"arm_2"` and values 5, 12, 7 (pull counts). Print the key with the highest count using `max(arm_counts, key=arm_counts.get)`.
2. Start with `Q = {}`. Set `Q[("state_A", "left")] = 0.3` and `Q[("state_A", "right")] = 0.7`. Print all key-value pairs with a `for` loop.
3. Write a function `best_action(Q, state, actions)` that returns the action with the highest Q-value for a given state (use `max(actions, key=lambda a: Q.get((state, a), 0.0))`).

{{< pyrepl code="Q = {}\nQ[(('s0', 'left'))] = 0.3\nQ[(('s0', 'right'))] = 0.7\n# TODO: print best action for 's0'\n" height="200" >}}

---

### Common pitfall

**Key not found:** `dict[key]` raises a `KeyError` if the key does not exist. Use `dict.get(key, default)` to return a default value instead (e.g. `Q.get((s, a), 0.0)`). In RL, you often want unvisited state-action pairs to default to 0.

---

## Importing modules

Python's standard library and third-party packages are accessed via `import`. In RL you use `random`, `numpy`, `matplotlib`, and more.

```python
import random

x = random.random()        # float in [0, 1)
a = random.randint(0, 3)   # integer in [0, 3] inclusive
choice = random.choice([0, 1, 2, 3])   # random element from list
```

**In RL we use this when:** Epsilon-greedy exploration needs `random.random() < epsilon` to decide whether to explore. Seeding with `random.seed(42)` makes results reproducible.

```python
import random
random.seed(42)   # reproducible

epsilon = 0.1
if random.random() < epsilon:
    action = random.randint(0, 3)   # explore
else:
    action = 2                       # exploit (pretend this is greedy)
print("Action:", action)
```

---

### Practice

1. Set `random.seed(0)` and simulate flipping a fair coin 10 times (use `random.choice([0, 1])`). Print the 10 outcomes and count how many were 1.
2. Simulate epsilon-greedy: set `epsilon = 0.2`, run 100 steps, count how many times you explored (random) vs. exploited. Print the counts.
3. Use `random.randint(0, 4)` in a loop to simulate a random agent on a 5-action problem for 50 steps. Print how often each action was taken.

{{< pyrepl code="import random\nrandom.seed(42)\n# Simulate 10 coin flips\nflips = [random.choice([0, 1]) for _ in range(10)]\nprint(flips)\nprint('Heads:', flips.count(1))" height="200" >}}

---

### Common pitfall

**Forgetting to seed:** Without `random.seed(n)`, results are different every run. During debugging, always seed your random number generator so you can reproduce a specific outcome.

---

## Reading error messages

When your code has a bug, Python prints a **traceback**. Learning to read it saves hours.

**NameError — variable not defined:**

```python
pint("Hello")   # typo in function name
# NameError: name 'pint' is not defined
```

Fix: Check spelling. It should be `print`.

**IndexError — index out of range:**

```python
rewards = [0, 1, 2]
print(rewards[5])
# IndexError: list index out of range
```

Fix: The list has 3 items (indices 0–2). Index 5 does not exist.

**TypeError — wrong type:**

```python
gamma = "0.9"   # string, not float
result = gamma ** 2
# TypeError: unsupported operand type(s) for ** ...
```

Fix: Use `gamma = 0.9` (a float).

**How to read a traceback:** The last line tells you the error type and message. The line above it tells you *which line* in your code caused it.

---

### Practice (find the bug)

Each snippet has one bug. Read the error, find and fix it.

1. ```python
   rewardss = [1, 0, 1]
   print(rewards[0])
   ```
2. ```python
   gamma = 0.9
   discount = gamma + 2
   print(discount)   # expected: 0.81
   ```
3. ```python
   steps = 5
   for i in range(steps):
       print("Step " + i)   # TypeError
   ```

{{< pyrepl code="# Fix this code:\nrewards = [1, 0, 1]\nprint(rewards[3])  # bug: index out of range\n# Hint: list has 3 items, valid indices are 0, 1, 2" height="200" >}}

---

## Checkpoint (before you continue)

Try these mini-exercises to confirm you can combine what you have learned:

1. **Checkpoint 1:** Write a short script (about 10 lines) that: (a) sets a variable `steps = 5`, (b) uses a `for` loop to print `"Step 0"`, `"Step 1"`, … up to `"Step 4"`, and (c) uses an `if` to print `"Done"` only when the loop variable equals 4.
2. **Checkpoint 2:** Write a function `total_reward(rewards)` that takes a list of numbers (e.g. `[0, 0, 1]`) and returns their sum. Call it from a loop that runs 2 times with different lists and prints the result each time.

If you can do both without looking back, you are ready for the next section.

---

## Reading and writing simple scripts

A typical script: define some variables and functions at the top, then use them in a small “main” section. You can run the script from the terminal.

### Example

A script that “runs” 3 episodes and prints a dummy return for each:

```python
def run_episode(episode_id):
    # Simulate 3 steps with rewards 0, 0, 1
    rewards = [0, 0, 1]
    total = sum(rewards)
    return total

# Main
for ep in range(3):
    ret = run_episode(ep)
    print("Episode", ep, "return:", ret)
```

Save as `episodes.py` and run `python3 episodes.py`. You should see three lines with returns 1, 1, 1.

**In RL we use this when:** Real training scripts are longer, but the structure is the same: load config, create environment and agent, loop over episodes, and inside each episode loop over steps until done. You are practicing that structure.

---

### Practice

1. Modify the script so each “episode” has a different list of rewards (e.g. [1], [0, 1], [0, 0, 1]) and run it again.
2. Write a function `discounted_return(rewards, gamma)` that computes \\(r_0 + \\gamma r_1 + \\gamma^2 r_2 + \\cdots\\) for a list `rewards` and a float `gamma`. Use a loop; do not use NumPy. Test with `rewards = [0, 0, 1]` and `gamma = 0.9`; the result should be \\(0.81\\).

---

### Professor's hint

Test small first. Get one episode working, then put it in a loop. Get one function right, then combine them. This is how you will debug RL code later.

---

## Phase 0 done? Checklist

Before moving on, confirm:

- [ ] I can run a Python script from the terminal (`python3 script.py`).
- [ ] I understand variables and types (int, float, str, bool).
- [ ] I can write an `if`/`elif`/`else` and a `for` or `while` loop.
- [ ] I can define a function with `def` and call it; I know what `return` does.
- [ ] I completed at least one of the Checkpoint exercises above.

If all are checked, you have finished Phase 0.

---

## You are ready for the full Python prerequisite

You now know:

- How to run a Python script.
- Variables and basic types (int, float, str, bool).
- Conditionals (`if` / `elif` / `else`).
- Loops (`for`, `while`).
- Defining and calling functions and returning values.

Next step: go to **[Prerequisites — Python](../../prerequisites/python/)**. There you will learn data structures (lists, tuples, dicts, sets), classes and objects, list comprehensions, and more patterns used in every RL codebase. The exercises there assume you can already write the kind of small programs you practiced in this phase.

After that, continue with the [Learning path](../): Phase 1 (Math for RL) and Phase 2 (rest of prerequisites), then the curriculum.

Good luck.
