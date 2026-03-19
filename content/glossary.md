---
title: "RL Glossary"
description: "Alphabetical glossary of all reinforcement learning terms used in this curriculum. Each entry includes a definition, the chapter where it is introduced, and an example."
date: 2026-03-19T00:00:00Z
draft: false
weight: 99
tags: ["glossary", "reference", "RL terms", "definitions"]
keywords: ["RL glossary", "reinforcement learning terms", "RL definitions", "RL reference"]
---

Use this glossary when you encounter an unfamiliar term. Each entry gives a one-line definition, the curriculum chapter where it first appears, and a concrete example. Terms are listed alphabetically.

---

## A

**Action (a)**
One choice the agent makes at a given time step. Actions come from a set called the action space.
*Introduced:* Chapter 1. *Example:* In a gridworld, actions are {up, down, left, right}.

**Action space**
The complete set of actions available to the agent. Can be discrete (finite actions) or continuous (e.g. a steering angle between −30° and +30°).
*Introduced:* Chapter 3.

**Actor-critic**
A class of algorithms that maintain both a policy (actor) and a value function (critic). The critic evaluates the actor's actions; the actor updates its policy using the critic's signal.
*Introduced:* Chapter 35.

**Advantage function A(s,a)**
The difference between the Q-function and the value function: A(s,a) = Q(s,a) − V(s). Measures how much better action a is compared to the average action in state s.
*Introduced:* Chapter 35. *Example:* If V(s)=0.5 and Q(s,left)=0.8, then A(s,left)=0.3.

**Agent**
The learner that interacts with the environment. The agent observes states, chooses actions, and receives rewards.
*Introduced:* Chapter 1.

**AlphaZero**
A model-based RL algorithm by DeepMind that combines Monte Carlo Tree Search with a neural network to play board games at superhuman level. Trained entirely via self-play.
*Introduced:* Chapter 55.

---

## B

**Baseline (in policy gradients)**
A function b(s) subtracted from the return G in REINFORCE to reduce variance without biasing the gradient. The value function V(s) is a common choice.
*Introduced:* Chapter 33.

**Behavioral cloning (BC)**
Imitation learning by supervised learning: train a policy to mimic expert actions from a dataset of (state, action) pairs.
*Introduced:* Chapter 75.

**Bellman equation**
A recursive equation expressing the value of a state in terms of the immediate reward and the discounted value of successor states. Fundamental to dynamic programming and TD methods.
*Introduced:* Chapter 6. *Example:* V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V^π(s')].

**Bellman optimality equation**
Like the Bellman equation but for the optimal policy. Q*(s,a) = Σ_{s'} P(s'|s,a) [R + γ max_{a'} Q*(s',a')].
*Introduced:* Chapter 6.

**Bootstrapping**
Updating value estimates using other value estimates (rather than waiting for full returns). TD methods bootstrap; Monte Carlo does not.
*Introduced:* Chapter 12.

**Buffer (replay buffer)**
A memory storing past transitions (s, a, r, s', done). The agent samples mini-batches from the buffer for training, breaking temporal correlations.
*Introduced:* Chapter 24.

---

## C

**CartPole**
A classic RL benchmark: balance a pole on a cart by applying left/right forces. Commonly used to test DQN and policy gradient algorithms.
*Introduced:* Chapter 23 (Vol 3 supplement).

**Clipping (PPO)**
In PPO, the probability ratio r(θ) = π_θ(a|s)/π_old(a|s) is clipped to [1−ε, 1+ε] to prevent excessively large policy updates.
*Introduced:* Chapter 43.

**Continuing task**
A task with no natural termination (no terminal state). The agent interacts indefinitely. Requires discounting (γ < 1) to keep returns finite.
*Introduced:* Chapter 4.

**CQL (Conservative Q-Learning)**
An offline RL algorithm that adds a regularization term penalizing high Q-values for out-of-distribution actions, preventing overestimation.
*Introduced:* Chapter 72.

**Critic**
The value function component in actor-critic methods. Evaluates the current policy by estimating V(s) or Q(s,a).
*Introduced:* Chapter 35.

**Cumulative regret**
The total difference between the reward of the optimal arm and the rewards collected by the agent over all steps. A measure of how much the bandit algorithm "loses" to an oracle.
*Introduced:* Chapter 2.

---

## D

**DAgger (Dataset Aggregation)**
An imitation learning algorithm that iteratively queries the expert on states visited by the learned policy, reducing distribution shift.
*Introduced:* Chapter 76.

**Decision Transformer**
A model that frames RL as a sequence prediction problem: given return-to-go, state, and action history, predict the next action.
*Introduced:* Chapter 73.

**Deterministic policy**
A policy that maps each state to exactly one action: π(s) = a. Contrast with a stochastic policy.
*Introduced:* Chapter 31.

**Discount factor (γ, gamma)**
A number in [0, 1] that down-weights future rewards. The return is G = r_0 + γr_1 + γ²r_2 + ⋯.
*Introduced:* Chapter 4. *Example:* γ=0.9 means a reward 3 steps away is worth 0.9³ ≈ 0.73 of its face value.

**Distribution shift**
The mismatch between the distribution of states/actions seen during data collection and those encountered at test time. A key challenge in offline RL and behavioral cloning.
*Introduced:* Chapter 75.

**DPO (Direct Preference Optimization)**
An alternative to PPO-based RLHF for LLMs. Directly optimizes a classification objective on preference pairs without training a separate reward model.
*Introduced:* Chapter 99.

**DQN (Deep Q-Network)**
A value-based deep RL algorithm that uses a neural network to approximate Q(s,a), with experience replay and target networks for stability.
*Introduced:* Chapter 23.

**Dueling DQN**
A DQN variant that decomposes Q(s,a) = V(s) + A(s,a) using separate network streams for value and advantage.
*Introduced:* Chapter 26.

**Dynamic programming (DP)**
A family of algorithms (policy evaluation, policy iteration, value iteration) that compute optimal policies using a known model of the environment (transition probabilities and rewards).
*Introduced:* Chapter 7.

---

## E

**Eligibility trace**
A mechanism for assigning credit to states visited in the past. TD(λ) uses eligibility traces to interpolate between TD(0) and Monte Carlo.
*Introduced:* Chapter 17.

**Environment**
Everything external to the agent. It receives actions and returns observations (states) and rewards.
*Introduced:* Chapter 1.

**Episode**
One complete sequence of interactions from an initial state to a terminal state (or until a step limit).
*Introduced:* Chapter 1.

**Epsilon-greedy (ε-greedy)**
An exploration strategy: with probability ε choose a random action; with probability 1−ε choose the greedy action (argmax Q).
*Introduced:* Chapter 2. *Example:* ε=0.1 means 10% random, 90% greedy.

**Experience replay**
Storing past transitions in a replay buffer and sampling random mini-batches for training. Breaks temporal correlations and improves sample efficiency.
*Introduced:* Chapter 24.

**Exploitation**
Choosing the action believed to give the highest reward based on current knowledge.
*Introduced:* Chapter 2.

**Exploration**
Trying actions to gain information about the environment, even if they may not give the highest immediate reward.
*Introduced:* Chapter 2.

**Exploration–exploitation trade-off**
The tension between gathering new information (exploration) and using current knowledge to maximize reward (exploitation). Central to bandit and RL problems.
*Introduced:* Chapter 2.

---

## F

**Feature vector φ(s)**
A fixed-size numerical representation of a state used in linear function approximation: V(s) = w · φ(s).
*Introduced:* Chapter 21.

**Function approximation**
Using a parameterized function (linear model or neural network) to represent V(s) or Q(s,a) when the state space is too large for a table.
*Introduced:* Chapter 21.

---

## G

**GAE (Generalized Advantage Estimation)**
A method for computing advantage estimates that interpolates between 1-step TD and Monte Carlo returns using a λ parameter.
*Introduced:* Chapter 44. *Example:* λ=0 → one-step TD advantage; λ=1 → Monte Carlo advantage.

**GAIL (Generative Adversarial Imitation Learning)**
Imitation learning using a GAN-style discriminator to distinguish agent transitions from expert transitions. The discriminator's output becomes the reward signal.
*Introduced:* Chapter 77.

**Gamma (γ)**
See *Discount factor*.

**Greedy policy**
A policy that always selects the action with the highest estimated value (argmax). May fail to explore.
*Introduced:* Chapter 2.

**Gymnasium (Gym)**
A Python library (originally OpenAI Gym) providing standardized RL environments with a step/reset API.
*Introduced:* Phase 2 prerequisites.

---

## H

**Hard exploration problem**
Settings where rewards are extremely sparse or delayed, making it difficult for standard exploration strategies to find them. Requires dedicated exploration methods.
*Introduced:* Chapter 61.

**Horizon**
The number of steps over which the agent plans. Finite horizon: T steps. Infinite horizon with γ < 1: effectively finite.
*Introduced:* Chapter 4.

---

## I

**ICM (Intrinsic Curiosity Module)**
An exploration method that provides intrinsic rewards based on prediction error of the agent's own dynamics model. High prediction error = novel state = high intrinsic reward.
*Introduced:* Chapter 63.

**IQL (Independent Q-Learning)**
A multi-agent RL approach where each agent independently applies Q-learning, ignoring other agents. Simple but may not converge.
*Introduced:* Chapter 83.

**IRL (Inverse Reinforcement Learning)**
Learning a reward function from expert demonstrations. Inverse of the RL problem: given behavior, infer the reward that makes it optimal.
*Introduced:* Chapter 76.

---

## L

**Learning rate (α, alpha)**
The step size used in value or weight updates. Controls how much new information overwrites old estimates.
*Introduced:* Chapter 12. *Example:* Q(s,a) ← Q(s,a) + α * δ where α = 0.1.

---

## M

**MAML (Model-Agnostic Meta-Learning)**
A meta-learning algorithm that trains a model initialization θ such that a few gradient steps on a new task yield good performance.
*Introduced:* Chapter 69.

**Markov property**
The property that the future is independent of the past given the present state: P(S_{t+1}|S_t, A_t) = P(S_{t+1}|S_0,...,S_t, A_0,...,A_t).
*Introduced:* Chapter 3.

**MDP (Markov Decision Process)**
The formal framework for RL: a tuple (S, A, P, R, γ) where S=states, A=actions, P=transition probabilities, R=reward function, γ=discount.
*Introduced:* Chapter 3.

**Model (of the environment)**
A learned or given representation of environment dynamics: P(s'|s,a) and R(s,a). Used in model-based RL.
*Introduced:* Chapter 51.

**Model-based RL**
RL that uses or learns a model of the environment to plan or generate simulated experience.
*Introduced:* Chapter 51.

**Model-free RL**
RL that learns directly from interaction with the environment without using a model. Includes Q-learning, SARSA, DQN, PPO.
*Introduced:* Chapter 11.

**Monte Carlo (MC) methods**
RL methods that estimate value functions using complete episodes (full returns) rather than bootstrapping.
*Introduced:* Chapter 11.

**MCTS (Monte Carlo Tree Search)**
A planning algorithm that builds a search tree by simulating many rollouts and using statistics to guide tree expansion. Used in AlphaZero.
*Introduced:* Chapter 54.

**Multi-armed bandit**
A simplified RL problem with a single state and multiple actions (arms). The agent repeatedly chooses arms to maximize cumulative reward.
*Introduced:* Chapter 2.

---

## N

**n-step return**
A return that uses n actual rewards followed by a bootstrap estimate: G_{t:t+n} = r_t + γr_{t+1} + ⋯ + γ^n V(S_{t+n}).
*Introduced:* Chapter 17.

**Nash equilibrium**
In a multi-player game, a strategy profile where no player can benefit by unilaterally changing their strategy.
*Introduced:* Chapter 82.

**Neural network (NN)**
A parameterized function composed of layers of linear transforms and nonlinear activations. Used in deep RL to approximate V(s) or Q(s,a).
*Introduced:* Chapter 22.

---

## O

**Off-policy learning**
Learning from data generated by a different (behavior) policy than the policy being improved (target policy). Q-learning is off-policy.
*Introduced:* Chapter 14.

**Offline RL**
RL that learns from a fixed dataset of transitions without further interaction with the environment. Also called batch RL.
*Introduced:* Chapter 71.

**On-policy learning**
Learning from data generated by the current policy being improved. SARSA and PPO are on-policy.
*Introduced:* Chapter 13.

**Optimistic initialization**
Setting initial Q-values higher than the true values to encourage early exploration of all actions.
*Introduced:* Chapter 2 (Vol 1 supplement).

---

## P

**PER (Prioritized Experience Replay)**
A replay buffer variant that samples transitions with probability proportional to their TD error magnitude. High-error transitions are replayed more often.
*Introduced:* Chapter 27.

**Policy (π)**
The agent's strategy: a mapping from states to actions (deterministic) or to action probabilities (stochastic).
*Introduced:* Chapter 1.

**Policy evaluation**
Computing the value function V^π for a given policy π.
*Introduced:* Chapter 7.

**Policy gradient**
A class of RL algorithms that directly optimize the policy parameters by following the gradient of expected return.
*Introduced:* Chapter 32.

**Policy improvement**
Constructing a new policy π' that is at least as good as π by acting greedily with respect to Q^π.
*Introduced:* Chapter 8.

**Policy iteration**
An algorithm that alternates between policy evaluation and policy improvement until convergence to the optimal policy.
*Introduced:* Chapter 8.

**PPO (Proximal Policy Optimization)**
An on-policy policy gradient algorithm that clips the probability ratio to prevent large policy updates. Widely used for its simplicity and stability.
*Introduced:* Chapter 43.

---

## Q

**Q-function (Q(s,a) or action-value function)**
The expected discounted return when taking action a in state s and then following policy π: Q^π(s,a) = E[G_t | S_t=s, A_t=a, π].
*Introduced:* Chapter 5.

**Q-learning**
An off-policy TD control algorithm: Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') − Q(s,a)].
*Introduced:* Chapter 14.

**Q-table**
A table (or dictionary) storing Q(s,a) values for all state-action pairs. Used in tabular Q-learning.
*Introduced:* Chapter 14.

---

## R

**Rainbow**
A DQN variant that combines six improvements: Double DQN, Dueling DQN, PER, n-step returns, distributional RL, and NoisyNet.
*Introduced:* Chapter 29.

**REINFORCE**
The foundational policy gradient algorithm: θ ← θ + α * G_t * ∇log π(A_t|S_t;θ), where G_t is the full episode return.
*Introduced:* Chapter 32.

**Replay buffer**
See *Buffer*.

**Return (G)**
The total discounted reward from step t: G_t = r_t + γr_{t+1} + γ²r_{t+2} + ⋯. The agent's goal is to maximize the expected return.
*Introduced:* Chapter 1.

**Reward (R or r)**
The scalar feedback the agent receives after each action. The only learning signal in RL.
*Introduced:* Chapter 1.

**Reward hypothesis**
The hypothesis that all goals can be described as maximization of cumulative scalar reward.
*Introduced:* Chapter 4.

**RLHF (Reinforcement Learning from Human Feedback)**
Training an LLM or agent using a reward model learned from human preference data, then fine-tuning with PPO.
*Introduced:* Chapter 78 and Chapter 98.

**RND (Random Network Distillation)**
An exploration method that trains a predictor network to match a fixed random network. Prediction error serves as intrinsic reward.
*Introduced:* Chapter 64.

---

## S

**SAC (Soft Actor-Critic)**
An off-policy maximum-entropy RL algorithm that adds an entropy bonus to the reward, encouraging exploration and robustness.
*Introduced:* Chapter 47.

**SARSA**
An on-policy TD control algorithm: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)], where a' is the action actually taken in s'.
*Introduced:* Chapter 13.

**Semi-gradient TD**
A TD update for parameterized value functions that applies the gradient only to the current value estimate, not the target.
*Introduced:* Chapter 21.

**State (s or S_t)**
The agent's description of the current situation. Contains enough information for decision-making (under the Markov assumption).
*Introduced:* Chapter 1.

**Stochastic policy**
A policy that maps states to probability distributions over actions: π(a|s). Naturally handles exploration and adversarial settings.
*Introduced:* Chapter 31.

---

## T

**Target network**
A copy of the Q-network with parameters updated less frequently (or slowly). Used in DQN to stabilize TD targets.
*Introduced:* Chapter 25.

**TD error (δ)**
The difference between the TD target and the current value estimate: δ = r + γV(s') − V(s).
*Introduced:* Chapter 12.

**TD learning (Temporal Difference learning)**
A family of methods that update value estimates using bootstrapped targets (mixing actual rewards with estimated future values).
*Introduced:* Chapter 12.

**TD(0)**
The simplest TD method: update V(s) using the one-step return r + γV(s').
*Introduced:* Chapter 12.

**TD(λ)**
A TD method that uses eligibility traces to blend n-step returns. λ=0 gives TD(0); λ=1 gives Monte Carlo.
*Introduced:* Chapter 17.

**Trajectory**
A sequence of states, actions, and rewards from one episode: (s_0, a_0, r_0, s_1, a_1, r_1, …, s_T).
*Introduced:* Chapter 1.

**TRPO (Trust Region Policy Optimization)**
A policy gradient algorithm that enforces a KL divergence constraint between old and new policy, ensuring stable updates.
*Introduced:* Chapter 42.

---

## U

**UCB (Upper Confidence Bound)**
A bandit algorithm that selects actions based on their estimated mean plus an uncertainty bonus, balancing exploration and exploitation.
*Introduced:* Chapter 2 (Vol 1 supplement). *Formula:* UCB1: a = argmax_i [Q(i) + c√(ln t / N(i))].

---

## V

**Value function V(s)**
The expected return from state s under policy π: V^π(s) = E[G_t | S_t=s, π].
*Introduced:* Chapter 5.

**Value iteration**
A DP algorithm that iteratively applies the Bellman optimality operator to compute V* (and therefore π*).
*Introduced:* Chapter 9.

---

## W

**World model**
A learned model of the environment's dynamics: given (s, a), predict (s', r). Used in model-based RL to plan in imagination.
*Introduced:* Chapter 52.

---

*This glossary covers terms from Phases 0–5 and Volumes 1–10. If you encounter a term not listed here, check the chapter index in the [Course Outline](course-outline/) or use the site search.*
