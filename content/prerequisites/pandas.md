---
title: "Pandas"
description: "Pandas for RL: DataFrames, Series, grouping, and logging metrics."
date: 2026-03-10T00:00:00Z
weight: 30
draft: false
difficulty: 2
tags: ["Pandas", "DataFrames", "metrics", "prerequisites"]
keywords: ["Pandas for RL", "DataFrames", "Series", "grouping", "logging metrics"]
roadmap_icon: "database"
roadmap_color: "green"
roadmap_phase_label: "Phase 2 · Pandas"
---

Useful for logging training metrics (rewards per episode, loss curves), loading small datasets, and analyzing results. Many curriculum exercises ask you to "plot the sum of rewards per episode"—storing those in a DataFrame keeps things tidy and easy to export.

---

## Why Pandas matters for RL

- **DataFrame** — Tabular data: one row per episode or per step, columns like `episode`, `reward`, `length`, `loss`. Easy to filter, aggregate, and plot.
- **Series** — 1D data (e.g. reward per episode). Rolling mean, describe(), and plot.
- **I/O** — `to_csv`, `read_csv` for saving/loading runs and sharing results.
- **Grouping and aggregation** — Mean reward per run, per algorithm, or per seed.

---

## Core concepts with examples

### Building a DataFrame from lists

```python
import pandas as pd

episodes = list(range(10))
rewards = [1, 2, 3, 2, 4, 3, 5, 4, 6, 5]
lengths = [10, 12, 15, 11, 20, 14, 22, 18, 25, 21]

df = pd.DataFrame({
    "episode": episodes,
    "reward": rewards,
    "length": lengths
})
```

### Basic columns and selection

```python
# Add a column (e.g. running mean)
df["reward_smooth"] = df["reward"].rolling(window=3, min_periods=1).mean()

# Select rows where reward > 4
df[df["reward"] > 4]

# Select columns
df[["episode", "reward"]]
```

### Describe and summary stats

```python
df["reward"].mean()
df["reward"].std()
df.describe()   # count, mean, std, min, 25%, 50%, 75%, max per column
```

### Save and load

```python
df.to_csv("rewards.csv", index=False)
df_loaded = pd.read_csv("rewards.csv")
```

### Plotting from a DataFrame

```python
import matplotlib.pyplot as plt
df.plot(x="episode", y="reward", label="reward")
df.plot(x="episode", y="reward_smooth", label="smoothed", ax=plt.gca())
plt.xlabel("Episode")
plt.show()
```

---

## Exercises

**Exercise 1.** Create a DataFrame with columns `episode` (0 to 99) and `reward`, where `reward` is `10 + episode * 0.1 + noise` with `noise` from `np.random.randn(100)`. Add a column `reward_ma5` that is the 5-step moving average of `reward`. Use `rolling(5, min_periods=1).mean()`.

**Exercise 2.** From the DataFrame in Exercise 1, compute the mean and standard deviation of `reward` over all episodes. Then compute the mean of `reward` only for episodes 50–99 (second half). Use boolean indexing: `df[df["episode"] >= 50]["reward"].mean()`.

**Exercise 3.** Simulate 3 "runs" of 20 episodes each: for each run, store `run_id` (0, 1, or 2), `episode` (0–19), and a random `reward` (e.g. `np.random.rand()`). Build one long DataFrame with 60 rows. Then use `groupby("run_id")["reward"].mean()` to get the mean reward per run.

**Exercise 4.** Load (or create) a DataFrame with `episode` and `reward`. Write a function that returns the **episode index** at which the reward first exceeds a threshold (e.g. 5.0). If it never exceeds, return `None`. Use boolean indexing and `.idxmax()` or a loop.

**Exercise 5.** Export a DataFrame to CSV with `to_csv("metrics.csv", index=False)`. Then read it back with `read_csv`. Verify that the numeric columns are identical (e.g. `(df - df_loaded).sum().sum() == 0` if no rounding). Add a column `timestamp` with the current time using `pd.Timestamp.now()` and export again.

**Exercise 6.** Create a DataFrame with `episode` (0–49) and `reward` (e.g. random or linearly increasing). Use `df["reward"].rolling(10, min_periods=1).mean()` to add a smoothed column. Plot both raw and smoothed reward vs episode with matplotlib. **In RL:** This is how you visualize learning curves.

**Exercise 7.** Simulate 2 algorithms: "DQN" and "SARSA", each with 100 episodes and random rewards. Build a DataFrame with columns `algorithm`, `episode`, `reward`. Use `groupby("algorithm")["reward"].mean()` to compare average reward per algorithm. **In RL:** You will compare many algorithms this way.

**Exercise 8.** (Challenge) Load a CSV of episode rewards (or create one). Write a function that returns the **number of episodes** needed until the rolling mean reward (window=10) first exceeds a threshold. Use a loop over episode indices and check the rolling mean at each step.

---

## Professor's hints

- **In RL:** Store one row per episode (or per eval episode) with columns like `episode`, `reward`, `length`, `loss`. Avoid one row per step for long runs—DataFrames get huge and slow.
- Use `rolling(window, min_periods=1).mean()` so the first few episodes still have a value (no NaN). For publication plots, window=10 or 20 is common.
- Save runs with `to_csv(..., index=False)` so episode IDs are not written as a separate index column. When comparing runs, add a `run_id` or `seed` column before concatenating.

---

## Common pitfalls

- **Chained assignment:** Avoid `df["new_col"][df["reward"] > 5] = 0`; it can trigger a warning and sometimes modify a copy. Use `df.loc[df["reward"] > 5, "new_col"] = 0` or assign a full column.
- **Reading CSV with wrong types:** After `read_csv`, check dtypes. Integer columns are usually fine; if episode IDs became floats, use `astype(int)`. Timestamps may need `pd.to_datetime`.
- **Plotting without grouping:** If you have multiple runs in one DataFrame, plot each run separately (e.g. loop over `run_id`) or use seaborn so curves do not get mixed up.

---

**Docs:** [pandas.pydata.org/docs](https://pandas.pydata.org/docs/).
