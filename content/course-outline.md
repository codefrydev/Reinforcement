---
title: "Course Outline"
description: "Full course outline in basic-to-advanced order. Every topic with links to curriculum, prerequisites, and learning path."
date: 2026-03-10T00:00:00Z
draft: false
weight: 1
tags: ["course outline", "syllabus", "reinforcement learning", "curriculum"]
keywords: ["course outline", "RL syllabus", "basic to advanced", "curriculum order"]
---

This page lists every topic in the intended order: from welcome and bandits through MDPs, dynamic programming, Monte Carlo, temporal difference, approximation methods, projects, and appendix. Follow this outline for a clear basic-to-advanced path. Each item links to the relevant curriculum chapter, prerequisite, or dedicated page.

---

## Welcome

| Topic | Where to find it |
|-------|------------------|
| [Introduction](../) | [Home](../) |
| Course Outline and Big Picture | This page |
| [Where to get the Code](../where-to-get-the-code/) | Dedicated page |
| [How to Succeed in this Course](../how-to-succeed/) | Dedicated page |

---

## Warmup — Multi-Armed Bandit

| Topic | Where to find it |
|-------|------------------|
| Section Introduction: The Explore-Exploit Dilemma | [Chapter 2: Multi-Armed Bandits](../curriculum/volume-01/chapter-02/) |
| Applications of the Explore-Exploit Dilemma | [Chapter 2](../curriculum/volume-01/chapter-02/) |
| Epsilon-Greedy Theory | [Chapter 2](../curriculum/volume-01/chapter-02/) |
| Calculating a Sample Mean (pt 1) | [Math for RL: Probability](../math-for-rl/probability/) |
| Epsilon-Greedy Beginner's Exercise Prompt | [Chapter 2](../curriculum/volume-01/chapter-02/) |
| Designing Your Bandit Program | [Chapter 2](../curriculum/volume-01/chapter-02/) |
| Epsilon-Greedy in Code | [Chapter 2](../curriculum/volume-01/chapter-02/) |
| Comparing Different Epsilons | [Chapter 2](../curriculum/volume-01/chapter-02/) |
| Optimistic Initial Values Theory | [Chapter 2](../curriculum/volume-01/chapter-02/) (hints); [Bandits: Optimistic Initial Values](../curriculum/volume-01/bandits-optimistic-initial-values/) |
| Optimistic Initial Values Beginner's Exercise Prompt | [Bandits: Optimistic Initial Values](../curriculum/volume-01/bandits-optimistic-initial-values/) |
| Optimistic Initial Values Code | [Bandits: Optimistic Initial Values](../curriculum/volume-01/bandits-optimistic-initial-values/) |
| [UCB1 Theory](../curriculum/volume-01/bandits-ucb1/) | Dedicated page |
| UCB1 Beginner's Exercise Prompt | [Bandits: UCB1](../curriculum/volume-01/bandits-ucb1/) |
| UCB1 Code | [Bandits: UCB1](../curriculum/volume-01/bandits-ucb1/) |
| [Bayesian Bandits / Thompson Sampling Theory (pt 1)](../curriculum/volume-01/bandits-thompson-sampling/) | Dedicated page |
| Bayesian Bandits / Thompson Sampling Theory (pt 2) | [Bandits: Thompson Sampling](../curriculum/volume-01/bandits-thompson-sampling/) |
| Thompson Sampling Beginner's Exercise Prompt | [Bandits: Thompson Sampling](../curriculum/volume-01/bandits-thompson-sampling/) |
| Thompson Sampling Code | [Bandits: Thompson Sampling](../curriculum/volume-01/bandits-thompson-sampling/) |
| Thompson Sampling With Gaussian Reward Theory | [Bandits: Thompson Sampling](../curriculum/volume-01/bandits-thompson-sampling/) |
| Thompson Sampling With Gaussian Reward Code | [Bandits: Thompson Sampling](../curriculum/volume-01/bandits-thompson-sampling/) |
| Exercise on Gaussian Rewards | [Bandits: Thompson Sampling](../curriculum/volume-01/bandits-thompson-sampling/) |
| [Why don't we just use a library?](../curriculum/volume-01/bandits-why-not-library/) | Dedicated page |
| [Nonstationary Bandits](../curriculum/volume-01/bandits-nonstationary/) | Dedicated page |
| Bandit Summary, Real Data, and Online Learning | [Chapter 2](../curriculum/volume-01/chapter-02/); [Bandits: Nonstationary](../curriculum/volume-01/bandits-nonstationary/) |
| (Optional) Alternative Bandit Designs | [Chapter 2](../curriculum/volume-01/chapter-02/) |

---

## High-Level Overview of Reinforcement Learning

| Topic | Where to find it |
|-------|------------------|
| [What is Reinforcement Learning?](../curriculum/volume-01/chapter-01/) | Chapter 1 |
| From Bandits to Full Reinforcement Learning | [Chapter 1](../curriculum/volume-01/chapter-01/), [Chapter 2](../curriculum/volume-01/chapter-02/) |
| [Markov Decision Processes](../curriculum/volume-01/chapter-03/) | Chapter 3 |

---

## MDP Section

| Topic | Where to find it |
|-------|------------------|
| MDP Section Introduction | [Chapter 3: MDPs](../curriculum/volume-01/chapter-03/) |
| [Gridworld](../curriculum/volume-01/gridworld/) | Dedicated page |
| [Choosing Rewards](../curriculum/volume-01/choosing-rewards/) | Dedicated page |
| The Markov Property | [Chapter 3](../curriculum/volume-01/chapter-03/) |
| Markov Decision Processes (MDPs) | [Chapter 3](../curriculum/volume-01/chapter-03/) |
| Future Rewards | [Chapter 4: Reward Hypothesis](../curriculum/volume-01/chapter-04/), [Chapter 5: Value Functions](../curriculum/volume-01/chapter-05/) |
| [Value Functions](../curriculum/volume-01/chapter-05/) | Chapter 5 |
| The Bellman Equation (pt 1–3) | [Chapter 6: The Bellman Equations](../curriculum/volume-01/chapter-06/) |
| Bellman Examples | [Chapter 6](../curriculum/volume-01/chapter-06/) |
| Optimal Policy and Optimal Value Function (pt 1–2) | [Chapter 6](../curriculum/volume-01/chapter-06/) |
| MDP Summary | [Chapter 3](../curriculum/volume-01/chapter-03/) – [Chapter 6](../curriculum/volume-01/chapter-06/) |

---

## Dynamic Programming

| Topic | Where to find it |
|-------|------------------|
| Dynamic Programming Section Introduction | [Volume 1](../curriculum/volume-01/) |
| [Iterative Policy Evaluation](../curriculum/volume-01/chapter-07/) | Chapter 7 |
| Designing Your RL Program | [Chapter 7](../curriculum/volume-01/chapter-07/) |
| [Gridworld in Code](../curriculum/volume-01/dp-gridworld-in-code/) | Dedicated page |
| [Iterative Policy Evaluation in Code](../curriculum/volume-01/dp-gridworld-in-code/) | Dedicated page |
| [Windy Gridworld](../curriculum/volume-01/windy-gridworld/) | Dedicated page |
| Iterative Policy Evaluation for Windy Gridworld | [Windy Gridworld](../curriculum/volume-01/windy-gridworld/) |
| Policy Improvement | [Chapter 8: Policy Iteration](../curriculum/volume-01/chapter-08/) |
| [Policy Iteration](../curriculum/volume-01/chapter-08/) | Chapter 8 |
| Policy Iteration in Code | [Chapter 8](../curriculum/volume-01/chapter-08/); [DP code walkthrough](../curriculum/volume-01/dp-gridworld-in-code/) |
| Policy Iteration in Windy Gridworld | [Windy Gridworld](../curriculum/volume-01/windy-gridworld/) |
| [Value Iteration](../curriculum/volume-01/chapter-09/) | Chapter 9 |
| Value Iteration in Code | [Chapter 9](../curriculum/volume-01/chapter-09/); [DP code walkthrough](../curriculum/volume-01/dp-gridworld-in-code/) |
| Dynamic Programming Summary | [Chapter 10: Limitations of DP](../curriculum/volume-01/chapter-10/) |

---

## Monte Carlo

| Topic | Where to find it |
|-------|------------------|
| [Monte Carlo Intro](../curriculum/volume-02/chapter-01/) | Chapter 11 |
| Monte Carlo Policy Evaluation | [Chapter 11](../curriculum/volume-02/chapter-01/) |
| [Monte Carlo Policy Evaluation in Code](../curriculum/volume-02/monte-carlo-in-code/) | Dedicated page |
| Monte Carlo Control | [Chapter 11](../curriculum/volume-02/chapter-01/) |
| Monte Carlo Control in Code | [Monte Carlo in Code](../curriculum/volume-02/monte-carlo-in-code/) |
| Monte Carlo Control without Exploring Starts | [Chapter 11](../curriculum/volume-02/chapter-01/); [Monte Carlo in Code](../curriculum/volume-02/monte-carlo-in-code/) |
| Monte Carlo Control without Exploring Starts in Code | [Monte Carlo in Code](../curriculum/volume-02/monte-carlo-in-code/) |
| Monte Carlo Summary | [Chapter 11](../curriculum/volume-02/chapter-01/) |

---

## Temporal Difference Learning

| Topic | Where to find it |
|-------|------------------|
| [Temporal Difference Introduction](../curriculum/volume-02/chapter-02/) | Chapter 12 |
| TD(0) Prediction | [Chapter 12](../curriculum/volume-02/chapter-02/) |
| [TD(0) Prediction in Code](../curriculum/volume-02/td-sarsa-q-in-code/) | Dedicated page |
| [SARSA](../curriculum/volume-02/chapter-03/) | Chapter 13 |
| SARSA in Code | [TD, SARSA, Q-Learning in Code](../curriculum/volume-02/td-sarsa-q-in-code/) |
| [Q-Learning](../curriculum/volume-02/chapter-04/) | Chapter 14 |
| Q-Learning in Code | [TD, SARSA, Q-Learning in Code](../curriculum/volume-02/td-sarsa-q-in-code/) |
| TD Learning Section Summary | [Chapter 12](../curriculum/volume-02/chapter-02/) – [Chapter 14](../curriculum/volume-02/chapter-04/) |

---

## Approximation Methods

| Topic | Where to find it |
|-------|------------------|
| Approximation Methods Section Introduction | [Volume 3](../curriculum/volume-03/) |
| [Linear Models for Reinforcement Learning](../curriculum/volume-03/chapter-01/) | Chapter 21 |
| [Feature Engineering](../curriculum/volume-03/feature-engineering/) | Dedicated page |
| Approximation Methods for Prediction | [Chapter 21](../curriculum/volume-03/chapter-01/) |
| Approximation Methods for Prediction Code | [Chapter 21](../curriculum/volume-03/chapter-01/) |
| Approximation Methods for Control | [Chapter 22](../curriculum/volume-03/chapter-02/) – [Chapter 30](../curriculum/volume-03/chapter-10/) |
| Approximation Methods for Control Code | [Volume 3](../curriculum/volume-03/) |
| [CartPole](../curriculum/volume-03/cartpole/) | Dedicated page |
| CartPole Code | [CartPole](../curriculum/volume-03/cartpole/) |
| Approximation Methods Exercise | [Volume 3](../curriculum/volume-03/) chapters |
| Approximation Methods Section Summary | [Volume 3](../curriculum/volume-03/) |

---

## Interlude: Common Beginner Questions

| Topic | Where to find it |
|-------|------------------|
| [This Course vs. RL Book: What's the Difference?](../course-vs-book/) | Dedicated page |
| [Stock Trading Project with Reinforcement Learning](../stock-trading/) | Dedicated section |
| Beginners, halt! Stop here if you skipped ahead | [Stock Trading intro](../stock-trading/) |
| Stock Trading Project Section Introduction | [Stock Trading](../stock-trading/) |
| Data and Environment | [Stock Trading: Data and Environment](../stock-trading/#data-and-environment) |
| How to Model Q for Q-Learning | [Stock Trading: How to Model Q](../stock-trading/#how-to-model-q-for-q-learning) |
| Design of the Program | [Stock Trading: Design](../stock-trading/#design-of-the-program) |
| Code pt 1–4 | [Stock Trading](../stock-trading/#code) |
| Stock Trading Project Discussion | [Stock Trading](../stock-trading/#discussion) |

---

## Appendix / FAQ

| Topic | Where to find it |
|-------|------------------|
| [What is the Appendix?](../appendix/) | [Appendix index](../appendix/) |
| [Setting Up Your Environment](../appendix/setting-up-environment/) | Dedicated page |
| Pre-Installation Check | [Setting Up Your Environment](../appendix/setting-up-environment/) |
| [Anaconda Environment Setup](../appendix/anaconda-setup/) | Dedicated page |
| How to install Numpy, Scipy, Matplotlib, Pandas, IPython, Theano, TensorFlow | [Installing Libraries](../appendix/installing-libraries/) |
| [How to Code by Yourself (part 1)](../appendix/how-to-code-by-yourself-1/) | Dedicated page |
| [How to Code by Yourself (part 2)](../appendix/how-to-code-by-yourself-2/) | Dedicated page |
| Proof that using Jupyter Notebook is the same as not using it | [Appendix](../appendix/) |
| Python 2 vs Python 3 | [Prerequisites: Python](../prerequisites/python/) |
| [Effective Learning Strategies](../appendix/effective-learning-strategies/) | Dedicated page |
| [How to Succeed in this Course (Long Version)](../appendix/how-to-succeed-long/) | Dedicated page |
| [Is this for Beginners or Experts? Academic or Practical? Pace](../appendix/beginners-or-experts/) | Dedicated page |
| [Machine Learning and AI Prerequisite Roadmap (pt 1–2)](../appendix/prerequisite-roadmap/) | Dedicated page |

---

## Part 2 — Advanced (Volumes 4–10)

After the topics above, the curriculum continues with 70 more chapters in order:

| Volume | Topics |
|--------|--------|
| [Volume 4: Policy Gradients](../curriculum/volume-04/) | Policy-based methods, REINFORCE, actor-critic, A2C, A3C, DDPG, TD3 (Ch 31–40) |
| [Volume 5: Advanced Policy Optimization](../curriculum/volume-05/) | TRPO, PPO, SAC, hyperparameter tuning (Ch 41–50) |
| [Volume 6: Model-Based RL & Planning](../curriculum/volume-06/) | World models, MCTS, AlphaZero, Dreamer, MBPO, PETS (Ch 51–60) |
| [Volume 7: Exploration and Meta-Learning](../curriculum/volume-07/) | Hard exploration, intrinsic motivation, RND, Go-Explore, MAML, RL² (Ch 61–70) |
| [Volume 8: Offline RL & Imitation Learning](../curriculum/volume-08/) | CQL, Decision Transformers, behavioral cloning, IRL, GAIL, RLHF (Ch 71–80) |
| [Volume 9: Multi-Agent RL (MARL)](../curriculum/volume-09/) | Game theory, IQL, CTDE, MADDPG, VDN, QMIX, MAPPO (Ch 81–90) |
| [Volume 10: Real-World RL, Safety & LLMs](../curriculum/volume-10/) | Robotics, safe RL, trading, recommenders, RLHF for LLMs, evaluation (Ch 91–100) |

See the full [Curriculum](../curriculum/) for all 100 chapters.
