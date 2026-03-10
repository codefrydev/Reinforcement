---
title: "RL Framework"
description: "Agent, environment, state, action, reward, Markov property, exploration-exploitation, and discount factor — with explanations."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["RL framework", "agent", "environment", "MDP", "exploration", "preliminary"]
keywords: ["reinforcement learning framework", "agent environment", "state action reward", "Markov", "discount factor"]
---

This page covers the core RL framework you need for the preliminary assessment: the four main components, the Markov property, exploration vs exploitation, and the discount factor. [Back to Preliminary](../).

---

## Why this matters for RL

Every RL problem is defined by who acts (agent), what they interact with (environment), what they observe (state), what they can do (actions), and what feedback they get (reward). The Markov property and the discount factor shape how we define value functions and algorithms. Exploration vs exploitation is the central tension in learning from experience.

### Learning objectives

Define agent, environment, action, reward, and state; state the Markov property and why it matters; give an example of the exploration-exploitation dilemma; explain \\(\gamma=0\\) vs \\(\gamma=1\\).

---

## Core concepts

- Agent: The learner/decision maker. Environment: Everything outside the agent it interacts with. Action: A choice the agent can make. Reward: A scalar feedback signal (immediate desirability). State: A representation of the current situation the agent uses to choose actions.
- Markov property: The future is independent of the past given the present state. So the state summarizes all relevant history.
- Exploration vs exploitation: Exploit = use what you believe is best; explore = try something else to get more information. We must balance them.
- Discount factor \\(\gamma\\): Weights future rewards. \\(\gamma=0\\): only immediate reward matters. \\(\gamma=1\\): future rewards count equally (can lead to infinite returns in continuing tasks).

**Illustration (discount factor):** For a sequence of rewards \\([0, 0, 1]\\), the return from step 0 is \\(0 + 0.9\\cdot 0 + 0.81\\cdot 1 = 0.81\\) when \\(\gamma=0.9\\). The chart below shows how the return from step 0 changes as we include more steps (1, 2, 3 steps).

{{< chart type="line" palette="return" title="Cumulative discounted return (γ=0.9)" labels="1 step, 2 steps, 3 steps" data="0, 0, 0.81" xLabel="Step" yLabel="Return" >}}

---

## Worked problems (with explanations)

### 1. Four components and state (Q10)

Q: Define the four main components of a reinforcement learning system: agent, environment, action, reward. Also, what is a state?

{{< collapse summary="Answer and explanation" >}}
- Agent: The learner/decision maker; the entity that selects actions.
- Environment: Everything outside the agent that it interacts with; it responds to actions with new states and rewards.
- Action: A move or decision the agent can make at a time step (from the action space).
- Reward: A scalar feedback signal from the environment indicating the immediate desirability of the current state or state-action transition.
- State: A representation of the current situation (provided by the environment or derived from observations) that the agent uses to decide which action to take. It should capture all information relevant to future outcomes (when the Markov property holds).

### Explanation

These five terms are the vocabulary of RL. The agent observes a state, takes an action, receives a reward and a next state from the environment, and repeats. Every algorithm (value-based, policy-based, model-based) is built on this loop. The state is the “summary” that we use for value functions and policies (e.g. \\(V(s)\\), \\(\pi(a|s)\\)).
{{< /collapse >}}

---

### 2. Markov property (Q11)

Q: What is the Markov property in the context of RL? Why is it important?

{{< collapse summary="Answer and explanation" >}}
The Markov property states that the future is independent of the past given the present state. In other words, the current state contains all the information needed to predict future states and rewards; the history of states and actions before the current state does not add any extra information.

### Why it's important

It allows us to model the problem as a Markov Decision Process (MDP). Then the value function and optimal policy depend only on the current state, not on the full history. That makes the problem tractable: we don’t need to condition on infinitely long histories, and algorithms like value iteration and Q-learning are well defined. When the state is not fully observed (partial observability), we may need to use history or a belief state, but the ideal is still to have a state that is Markov.
{{< /collapse >}}

---

### 3. Exploration vs exploitation (Q12)

Q: Give a real-world example of the exploration-exploitation dilemma and explain why it's challenging.

{{< collapse summary="Answer and explanation" >}}
### Example

Choosing a restaurant. Exploitation means going to a place you already know and like. Exploration means trying a new place that might be better (or worse). If you only exploit, you may never find a better option. If you only explore, you waste meals on bad choices. The challenge is balancing short-term satisfaction (exploit) with long-term discovery (explore).

### In RL

The agent must exploit what it believes is best to get high reward, but also explore to improve its estimates and discover better actions. Too little exploration leads to suboptimal policies; too much leads to slow learning or excessive risk. Algorithms like ε-greedy, UCB, and Thompson sampling are designed to manage this trade-off.
{{< /collapse >}}

---

### 4. Discount factor (Q13)

Q: What is the purpose of a discount factor \\(\gamma\\) in RL? What happens when \\(\gamma=0\\) and when \\(\gamma=1\\) (in continuing tasks)?

{{< collapse summary="Answer and explanation" >}}
The discount factor \\(\gamma \in [0, 1]\\) determines the present value of future rewards. The return is \\(G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots\\). So \\(\gamma\\) makes the sum finite in continuing tasks (no terminal state) and encodes time preference / uncertainty about the future.

- \\(\gamma=0\\): Only the immediate reward matters. The agent is “myopic”; it ignores future consequences. Useful for very short-term goals but usually not desirable for long-term planning.
- \\(\gamma=1\\): Future rewards count as much as immediate ones. In *continuing* tasks, the return can be infinite (e.g. constant positive reward forever), so we need either episodic tasks or a different formulation. In episodic tasks, \\(\gamma=1\\) is often used so that total undiscounted return is the objective.

### Explanation

In practice we usually choose \\(\gamma\\) close to 1 (e.g. 0.99) so the agent cares about the long run but the sum still converges. The Bellman equations and all value-based algorithms depend on \\(\gamma\\).
{{< /collapse >}}

---

## Toy example: grid world

Consider a 2×2 grid. States: cells (1,1), (1,2), (2,1), (2,2). Actions: up, down, left, right (with boundaries blocking moves). Reward: +1 for reaching a goal cell, 0 otherwise. Agent: chooses action each step. Environment: returns next state and reward. If the next state and reward depend only on current state and action (and maybe a fixed transition noise), the state is Markov. The discount factor \\(\gamma\\) then weights how much we care about delayed reward (reaching the goal in few vs many steps).

---

## Professor's hints

- When you see “MDP,” think: states, actions, rewards, transition dynamics, and (often) \\(\gamma\\). The agent doesn’t need to know the dynamics; it can learn from experience.
- Exploration is necessary when we don’t know the best action. Once we know (or think we know) the best action, we can exploit.
- In continuing tasks, \\(\gamma < 1\\) is usually required so that the infinite sum of discounted rewards is finite.

---

## Common pitfalls

- Confusing reward with return: Reward is per-step; return is the (discounted) sum of future rewards. Value functions are expectations of return, not of a single reward.
- Assuming full observability: The Markov property is about the *state*, not the raw observation. If the state is “what the agent sees” and that omits important information, the state is not Markov and we may need POMDPs or history.
- Using \\(\gamma=1\\) in continuing tasks without care: The return can be infinite; algorithms that assume bounded returns may break. Use \\(\gamma < 1\\) or average reward formulation.
