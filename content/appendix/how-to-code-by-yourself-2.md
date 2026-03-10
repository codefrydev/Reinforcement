---
title: "How to Code by Yourself (part 2)"
description: "Practice habits, when to peek at the solution, and building a coding routine."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["FAQ", "coding", "learning", "practice"]
keywords: ["coding practice", "solutions", "debugging", "habit"]
---

**Learning objectives**

- Build a habit of coding a little every day (or every session).
- Know when to look at the worked solution and how to use it without copying.
- Use print statements and small tests to debug.

## Practice regularly

- **Short sessions:** Even 20–30 minutes of coding (one small step: e.g. "get the bandit environment returning rewards") counts. Consistency beats rare long sessions.
- **One exercise at a time:** Finish (or get seriously stuck on) one exercise before jumping to the next. The curriculum is ordered so that skills build.
- **Re-do later:** After reading the solution, close it and re-implement the same exercise a few days later. You will remember the idea but have to write the code again—that strengthens retention.

## When to look at the solution

- **After a real attempt:** Try for at least 15–30 minutes. Write something—even if it is wrong. Then open the solution.
- **When stuck on one bug:** If you have one specific bug (e.g. "my value function is all zeros"), try to fix it with print statements or a minimal example. If you are still stuck after 15 minutes, look at how the solution handles that part.
- **Do not just copy:** After reading the solution, close it and type the code yourself. You will understand it better than if you copy-paste.

## Proof that using Jupyter Notebook is the same as not using it

You can do all exercises in **Jupyter** (run cells, plot inline) or in **plain Python scripts** (run with `python script.py`). The algorithms and math are the same. Jupyter is convenient for plotting and trying small pieces; scripts are convenient for running full experiments and version control. Use whichever you prefer. The curriculum does not require one or the other.

## Python 2 vs Python 3

This curriculum uses **Python 3** (3.8+). Python 2 is end-of-life. If you see Python 2 syntax elsewhere (e.g. `print x`), translate to Python 3 (`print(x)`). All code in this course is Python 3.

See [How to Code by Yourself (part 1)](../how-to-code-by-yourself-1/) for reading specs and breaking down tasks, and [Effective Learning Strategies](../effective-learning-strategies/) for broader study habits.
