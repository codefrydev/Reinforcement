---
title: "How to Code by Yourself (part 1)"
description: "Building independence in coding—reading specs, breaking problems down, and trying small steps."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["FAQ", "coding", "learning", "beginners"]
keywords: ["how to code", "coding by yourself", "programming", "learning"]
---

**Learning objectives**

- Read an exercise or spec and identify the inputs, outputs, and steps.
- Break a coding task into small, testable steps.
- Use documentation and error messages to fix issues without giving up.

## Why "by yourself" matters

The curriculum gives you **exercises with worked solutions**. The goal is not to copy the solution but to **try first**, then check. Coding by yourself—even when you get stuck—builds the skill to implement algorithms and debug them later in real projects or research. This part focuses on **reading and planning**.

## Read the spec carefully

Before writing code:

1. **What is the input?** (e.g. number of arms, step count, environment.)
2. **What is the output?** (e.g. a plot, a value function, a policy.)
3. **What are the exact rules?** (e.g. "first-visit MC," "epsilon-greedy with ε=0.1," "stop when max |ΔV| < 1e-4.")

Underline or list the key phrases. Many bugs come from misreading one detail (e.g. first-visit vs. every-visit, or updating terminal states when you should not).

## Break the problem into steps

Do not write the whole program at once. For example, for a bandit:

1. Implement the environment (sample reward from each arm).
2. Implement the agent (choose action, update Q).
3. Run one step; then run many steps; then many runs and average.
4. Add plotting.

Test each step (e.g. "after 10 pulls, do I have 10 rewards? Are my Q values updating?"). If something breaks, you know which step is wrong.

## Use documentation and errors

- **Documentation:** For NumPy, Python, Gym, etc., use the official docs or a quick web search ("numpy random choice"). Look at function signatures and examples.
- **Error messages:** Read the traceback. The last line often tells you the error type (e.g. IndexError, KeyError). The line number points to where it happened. Fix that first; sometimes one fix resolves several errors.

See [How to Code by Yourself (part 2)](how-to-code-by-yourself-2/) for practice habits and when to look at the solution.
