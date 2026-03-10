# Reinforcement Learning Curriculum

**Main goal: learn reinforcement learning from zero, ground up.**

This project is a full curriculum and website for studying reinforcement learning from the very beginning—no prior RL experience required. It takes you from mathematical foundations and multi-armed bandits through Markov decision processes, dynamic programming, Monte Carlo and temporal-difference methods, function approximation, and on to advanced topics (policy gradients, model-based RL, offline RL, multi-agent RL, and more).

## What’s inside

- **100 chapters** across 10 volumes, ordered from basics to advanced
- **Learning path for absolute beginners** — from zero programming to building RL systems
- **Prerequisites** — Python, NumPy, Pandas, Matplotlib, PyTorch, and related tools
- **Math for RL** — probability, linear algebra, calculus as needed for the material
- **Readiness assessments** — preliminary quiz and phase quizzes (math, foundations, deep RL)
- **Worked solutions** — one exercise per chapter with collapsible solutions
- **Course outline** — full syllabus with links to every topic in order

## How to use it

1. **New to RL?** Start with the [learning path](content/learning-path/) (from zero to RL step by step).
2. **Check readiness.** Take the [preliminary assessment](content/preliminary/) and phase quizzes if you want to see where you stand.
3. **Follow the curriculum.** Go through [Volume 1](content/curriculum/volume-01/) (mathematical foundations, bandits, MDPs, DP) and continue in order through the volumes.
4. **Use the outline.** The [course outline](content/course-outline.md) lists every topic in basic-to-advanced order with links.

The content is written in Markdown under `content/`. The site is built with [Hugo](https://gohugo.io/) and the PaperMod theme.

## Run the site locally

```bash
# Install Hugo (if needed): https://gohugo.io/installation/
hugo server
```

Then open http://localhost:1313/ (or the URL Hugo prints). The site will live-reload as you edit content.

## Build for production

```bash
hugo
```

Output goes to the `public/` directory.

## Project layout (high level)

| Path | Purpose |
|------|---------|
| `content/` | All curriculum, learning path, prerequisites, assessments, and outline |
| `content/curriculum/` | 10 volumes, 100 chapters |
| `content/learning-path/` | Beginner path from zero to RL |
| `content/prerequisites/` | Python, NumPy, PyTorch, etc. |
| `content/math-for-rl/` | Probability, linear algebra, calculus |
| `layouts/` | Hugo layouts and shortcodes |
| `hugo.yaml` | Site configuration |

## About this project

Most of the content in this project is gathered from the internet. The initial layout and structure were created with the help of AI. This is an **initial phase**—the material will be refined as I learn more and more, so **expect lots and lots of changes** over time.

---

Good luck on your journey from zero to mastery in reinforcement learning.
