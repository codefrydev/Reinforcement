---
title: "OpenAI Gym / Gymnasium"
description: "Standard RL environments: reset, step, spaces, wrappers, and seeding."
date: 2026-03-10T00:00:00Z
weight: 70
draft: false
tags: ["Gym", "Gymnasium", "environments", "reset", "step", "prerequisites"]
keywords: ["OpenAI Gym", "Gymnasium", "RL environments", "reset step", "spaces", "wrappers"]
---

The curriculum uses Gym-style environments (e.g. Blackjack, Cliff Walking, CartPole, LunarLander). [Gymnasium](https://gymnasium.farama.org/) is the maintained fork of OpenAI Gym. The same API appears in many exercises: reset, step, observation and action spaces.

---

## Why Gym matters for RL

- **API** — `env.reset()` returns `(obs, info)`; `env.step(action)` returns `(obs, reward, terminated, truncated, info)`. Episodes run until `terminated or truncated`.
- **Spaces** — `env.observation_space` and `env.action_space` describe shape and type (Discrete, Box). You need them to build networks and to sample random actions.
- **Wrappers** — Record episode stats, normalize observations, stack frames, or limit time steps without changing the base env.
- **Seeding** — Reproducibility via `env.reset(seed=42)` and `env.action_space.seed(42)`.

---

## Core concepts with examples

### Basic loop: reset and step

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)
done = False
total_reward = 0
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
env.close()
print("Episode return:", total_reward)
```

### Inspecting spaces

```python
print(env.observation_space)   # Box(4,) for CartPole
print(env.action_space)        # Discrete(2)
# Sample actions
action = env.action_space.sample()
# For Box (continuous): low, high, shape
# env.observation_space.low, .high, .shape
```

### Multiple episodes

```python
n_episodes = 10
returns = []
for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    G = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        G += reward
    returns.append(G)
env.close()
print("Mean return:", sum(returns) / len(returns))
```

### Wrappers: record episode stats

```python
from gymnasium.wrappers import RecordEpisodeStatistics

env = gym.make("CartPole-v1")
env = RecordEpisodeStatistics(env)
obs, info = env.reset()
# ... run episode ...
# After step that ends episode, info may contain "episode": {"r": ..., "l": ...}
```

### Seeding for reproducibility

```python
env.reset(seed=0)
env.action_space.seed(0)
# Same sequence of random actions and (with a deterministic env) same trajectory
```

---

## Exercises

**Exercise 1.** Create a CartPole-v1 environment. Call `reset(seed=42)` and then take 10 random actions with `action_space.sample()`, calling `step` each time. Print the observation shape and the cumulative reward after 10 steps. Close the env.

**Exercise 2.** Run 100 episodes of CartPole with a **random policy** (sample action each step). Store the return (sum of rewards) for each episode in a list. Compute and print the mean and standard deviation of returns. Use a fixed seed for `reset` and `action_space` so the result is reproducible.

**Exercise 3.** Inspect the observation and action spaces of "CartPole-v1" and "LunarLander-v2" (or LunarLanderContinuous-v2). Print the type (Discrete/Box), shape, and for Box the low/high bounds. Write a short comment on how you would size the input and output layers of a neural network for each.

**Exercise 4.** Implement a simple **fixed policy** for CartPole: if the cart position (obs[0]) is positive, take action 1; else take action 0. Run 20 episodes with this policy and record the return for each. Report the mean return. (This policy is poor; the exercise is just to practice using a non-random policy.)

**Exercise 5.** Write a function `run_episode(env, policy, max_steps=500)` that runs one episode: reset, then loop step until terminated, truncated, or max_steps. The `policy` is a callable `policy(obs) -> action`. Return the list of (obs, action, reward) for each step and the total return. Test with a random policy and with the fixed policy from Exercise 4.

**Exercise 6.** Run 50 episodes of CartPole with a random policy. Store the **length** (number of steps) of each episode. Compute the mean and max length. **In RL:** Episode length is often reported alongside return; for CartPole, longer is better.

**Exercise 7.** Create Blackjack (e.g. `gym.make("Blackjack-v1")`). Run 10 episodes with a random policy (sample from `env.action_space`). Print the observation shape and the meaning of the first few components (player sum, dealer card, usable ace) from the docs. **In RL:** Blackjack is used in the curriculum for Monte Carlo prediction.

**Exercise 8.** (Challenge) Write a wrapper that **counts** the number of steps per episode and, when the episode ends, prints "Episode finished in N steps, return R". Use a class that holds `env`, overrides `step` to count and check `terminated or truncated`, and prints on done. **In RL:** Custom wrappers are used for logging, frame stacking, and reward shaping.

---

## Professor's hints

- **Always set `done = terminated or truncated`**; Gymnasium uses both flags. Ignoring `truncated` (e.g. time limit) can lead to wrong value estimates or infinite loops.
- **In RL:** Seed both `env.reset(seed=...)` and `env.action_space.seed(...)` so the environment and your random actions are reproducible. Do this once per run or once per episode depending on what you want to reproduce.
- Use `env.observation_space.shape` and `env.action_space.n` (for Discrete) to size your neural network. For Box, use `env.observation_space.shape[0]` for the state dimension.
- Call `env.close()` when you are done (e.g. after all episodes); some envs use resources that should be released.

---

## Common pitfalls

- **Using the old Gym API:** Old Gym used `done` and 4 return values. Gymnasium uses 5: `(obs, reward, terminated, truncated, info)`. Check the library version and docs.
- **Assuming obs is a numpy array:** It usually is, but some envs return dicts or other types. Check `type(obs)` and `obs.shape` before passing to a network.
- **Forgetting to handle truncation:** If you only check `terminated`, time-limited episodes may never "end" in your loop logic. Always use `done = terminated or truncated`.
- **Not seeding:** Without seeds, you cannot reproduce results or debug. Seed at the start of training and (if you want identical episodes) per episode.

---

**Docs:** [gymnasium.farama.org](https://gymnasium.farama.org/). Used in Chapters 11–12 (Blackjack), 13–16 (Cliff Walking), 23+ (CartPole, etc.).
