---
title: "Anaconda Environment Setup"
description: "Create and use a conda environment for the RL curriculum."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["Anaconda", "conda", "environment", "FAQ"]
keywords: ["Anaconda", "conda", "virtual environment", "setup"]
weight: 5
roadmap_icon: "terminal"
roadmap_color: "purple"
roadmap_phase_label: "Setup"
---

**Learning objectives**

- Create a dedicated conda environment for the curriculum.
- Install Python and key packages in that environment.
- Activate and use the environment for running exercises.

## Why use a conda environment?

A **conda environment** isolates the curriculum’s Python and packages from your system or other projects. You can use a specific Python version and install NumPy, PyTorch, Gym, etc. without affecting other work. If something breaks, you can recreate the environment.

## Steps

1. **Install Miniconda or Anaconda** (if not already installed). Miniconda is minimal; Anaconda includes more packages and an IDE. Download from [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html).

2. **Create an environment** (e.g. named `rl`):
   ```bash
   conda create -n rl python=3.10
   ```

3. **Activate the environment:**
   - Linux/macOS: `conda activate rl`
   - Windows: `conda activate rl` (in Anaconda Prompt or terminal that has conda in PATH)

4. **Install packages:**
   ```bash
   conda install numpy matplotlib
   pip install gym  # or pip install gymnasium
   pip install torch  # for PyTorch (or use conda install pytorch)
   ```
   See [Installing Libraries](installing-libraries/) for more options (TensorFlow, etc.).

5. **Run your scripts or Jupyter** from this environment so they use the correct Python and packages.

6. **Deactivate when done:** `conda deactivate`

## Tips

- To use this environment in Jupyter: `pip install ipykernel` then `python -m ipykernel install --user --name rl --display-name "Python (rl)"`. Then choose the "Python (rl)" kernel in Jupyter.
- To remove the environment later: `conda env remove -n rl`

See [Setting Up Your Environment](setting-up-environment/) for a pre-install check and [Installing Libraries](installing-libraries/) for package details.
