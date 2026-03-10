---
title: "Setting Up Your Environment"
description: "Pre-installation check and what you need to run the curriculum code and exercises."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["setup", "environment", "FAQ", "installation"]
keywords: ["environment setup", "pre-installation", "RL curriculum", "Python"]
---

**Learning objectives**

- Know what software you need (Python, libraries, optional IDE).
- Perform a pre-installation check so you are ready for the curriculum.

## Pre-Installation Check

Before diving into the curriculum, ensure you have:

1. **Python:** Version 3.8 or higher (3.9–3.11 recommended). Check with `python3 --version` or `python --version`.
2. **pip:** So you can install packages. Check with `pip --version` or `pip3 --version`.
3. **Optional but recommended:** A virtual environment (venv or conda) so curriculum dependencies do not conflict with other projects. See [Anaconda Setup](anaconda-setup/) for conda.
4. **Libraries used in the curriculum:** NumPy, Matplotlib, and (for later volumes) PyTorch or TensorFlow, and Gym/Gymnasium. See [Installing Libraries](installing-libraries/) for how to install them.

## What you need

- **For Volumes 1–2 (foundations, tabular methods):** Python, NumPy, Matplotlib. You can implement gridworld, bandits, Monte Carlo, and TD in plain Python + NumPy; plotting helps for learning curves.
- **For Volume 3+ (function approximation, deep RL):** PyTorch or TensorFlow, and Gym or Gymnasium for environments (CartPole, MountainCar, etc.).
- **Editor or IDE:** Any text editor or IDE (VS Code, PyCharm, etc.). Jupyter is optional; see the FAQ on "Proof that using Jupyter Notebook is the same as not using it" (you can use scripts or notebooks—both are fine).

## After setup

Once your environment is ready, take the [Preliminary assessment](../preliminary/) to see if you are ready for the curriculum, or follow the [Learning path](../learning-path/) from Phase 0 if you are new to programming.
