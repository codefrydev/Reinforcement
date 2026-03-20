---
title: "RL in Plain English"
description: "Reinforcement learning explained with everyday analogies — no math, no code. Read this before starting the curriculum."
date: 2026-03-19T00:00:00Z
draft: false
difficulty: 0
weight: 3
tags: ["RL intro", "plain English", "beginner", "analogies", "reinforcement learning"]
keywords: ["RL in plain English", "reinforcement learning for beginners", "RL analogies", "what is reinforcement learning", "no math RL"]
roadmap_icon: "book"
roadmap_color: "teal"
roadmap_phase_label: "RL in Plain English"
---

Before any formulas or code, let's build intuition. Every concept in RL has a natural everyday version. This page explains RL ideas using plain English and real-world situations. There is **no math and no code** here — just understanding.

---

## What is Reinforcement Learning?

**RL is learning by doing, with feedback.**

Think about how a child learns to walk:

1. The child tries to stand up. They wobble and fall — that hurts (negative feedback).
2. They try again, shifting their balance differently. Still falls, but less badly.
3. After hundreds of tries, they find a balance that works. Standing upright feels stable (positive feedback).
4. Gradually, they learn to take a step, then another.

Nobody gave the child a manual. Nobody showed them a diagram of muscle forces. They tried things, got feedback, and adjusted. **That is reinforcement learning.**

Another example: a dog learning tricks. You say "sit." The dog randomly sits. You give it a treat (reward). It learns: sitting when told = treat. A few dozen repetitions and the dog knows the trick. No one programmed the dog — it learned from **experience and reward signals**.

---

## The Five Core Ideas

### 1. Agent

The **agent** is the learner. It decides what to do. In the dog example, the dog is the agent. In RL, the agent could be:

- A robot deciding which way to move
- A computer program deciding which move to make in a chess game
- An algorithm deciding which ad to show you
- A trading system deciding whether to buy or sell a stock

The agent has no built-in knowledge about what's good or bad. It **discovers** what works through experience.

---

### 2. Environment

The **environment** is everything the agent interacts with. It responds to what the agent does. The floor is the dog's environment — it determines whether standing upright is stable. The chess board is the game AI's environment.

A key insight: **the agent doesn't control the environment — it only controls its own choices**. The environment might be unpredictable or partially hidden. That's what makes RL hard and interesting.

---

### 3. State

At any moment, the agent sees a **description of the situation** — that's the state. For a robot in a room, the state might be its position and which direction it's facing. For a chess player, the state is the current arrangement of pieces on the board.

Think of the state as "the snapshot the agent sees right now."

| Scenario | What the state looks like |
|---|---|
| Robot in a warehouse | Row 3, Column 7, facing north |
| Chess AI | All 64 squares and which piece is on each |
| Recommendation system | "User has watched 5 cooking videos today" |
| Stock trading bot | Current price, portfolio, last 10 days of prices |
| Hospital treatment | Patient age, diagnosis, current medications |

---

### 4. Action

The **action** is what the agent decides to do. From the state, the agent picks one action. Actions change the state (or at least affect it).

| Scenario | Possible actions |
|---|---|
| Robot | Move forward, turn left, turn right, stop |
| Chess AI | Move any legal piece to any legal square |
| Recommendation | Show video A, show video B, show video C |
| Stock trading | Buy 10 shares, sell 10 shares, hold |

**The agent's job is to learn which action to take in each state.**

---

### 5. Reward

The **reward** is the feedback signal. After taking an action, the agent receives a number — positive (good), negative (bad), or zero (neutral). This is the only "teaching signal" the agent gets.

Important: the reward is often **delayed**. A chess move might not pay off until 20 moves later. A diet change might not affect health for months. RL agents learn to plan for the future, not just chase immediate rewards.

| Scenario | Reward |
|---|---|
| Robot navigation | +1 when it reaches the goal; small penalty each step |
| Chess | +1 for winning, -1 for losing, 0 otherwise |
| Recommendation | +1 if user clicks and watches, 0 if they skip |
| Trading | Daily profit or loss |
| Healthcare | Patient health score after treatment |

---

## Policy: Your Strategy

A **policy** is the agent's strategy — the rule that decides what action to take in any situation. 

Think of it like a habit or a rule of thumb:

- "If I see a red light, I stop." (robot)
- "If I'm behind in chess, I trade pieces." (chess AI)
- "If the user watches cooking videos, show more cooking." (recommendation)

A policy can be **simple** ("always go right") or **complex** ("look at 1000 factors and pick the option that maximizes expected revenue"). RL learns the policy automatically from experience.

---

## Value: How Good Is This Situation?

The **value** of a situation answers: "If I'm here, how much reward can I expect to get from here onwards?"

Examples:

- Being one step from the goal = high value (reward is close)
- Being trapped in a dead end = low value (reward is far away or impossible)
- Having a winning board position in chess = high value
- Having a patient in early-stage disease = medium value (treatment is more effective earlier)

Value tells the agent whether it's doing well, not just in the moment, but **in the long run**. This is what separates RL from simple reflexes.

---

## Exploration vs. Exploitation: The Restaurant Dilemma

You are in a new city and hungry. You know one restaurant — it's decent. Should you go there again (safe choice) or try somewhere new (risky, might be better or worse)?

This is the **explore-exploit dilemma** — one of the most fundamental tensions in RL:

| Explore | Exploit |
|---|---|
| Try something new | Stick with what works |
| Risk getting something worse | Miss something better |
| Gain new information | Use current knowledge |

More examples:

- **Drug discovery:** Try a new compound (explore) or use the best known drug (exploit)?
- **Ad recommendations:** Show a new ad you're uncertain about (explore) or show the ad that has worked best (exploit)?
- **Robot in a maze:** Try a new path you haven't explored (explore) or follow the known path to the goal (exploit)?
- **Chess opening:** Try a novel opening (explore) or use your trusted strategy (exploit)?

A pure exploiter never learns anything new and gets stuck with a mediocre solution. A pure explorer never commits to a good solution and wastes time. **Good RL agents balance both.**

---

## Episodes and Steps

An **episode** is one complete run from start to finish — like one game, one trip, or one day.

- Chess: one game (start to checkmate or draw)
- Robot in a maze: start at the entrance, end when it reaches the exit or gets stuck
- Recommendation: one user session

A **step** is one interaction: the agent observes the state, takes an action, and gets a reward. An episode is made of many steps.

Some tasks never end (like managing a server or a long-running recommendation system). These are called **continuing tasks**.

---

## Discount Factor: Future Rewards Are Worth Less

Would you rather have £10 today or £10 next year? Most people prefer today. Future money is worth less because it's uncertain and delayed.

RL uses the same idea with the **discount factor** (usually called γ, "gamma"). A reward received two steps from now is worth `γ²` times what it would be worth now.

- γ = 1.0: future rewards count exactly as much as immediate rewards
- γ = 0.9: a reward in 3 steps is worth 0.9 × 0.9 × 0.9 = 0.729 of what it would be now
- γ = 0.0: only the immediate reward matters (extremely short-sighted)

In practice, γ is usually between 0.9 and 0.99. It keeps the math well-behaved (prevents infinite sums) and captures the intuition that immediate rewards are more certain.

---

## Putting It All Together

Here is the full RL loop described in plain English:

1. **The agent observes the current state** ("I am at position (2, 3) in the maze.")
2. **The agent picks an action** based on its current policy ("My policy says: go right.")
3. **The environment responds** ("You moved to (2, 4). The reward is 0.")
4. **The agent updates its knowledge** ("Going right from (2,3) wasn't great — I'll adjust.")
5. **Repeat** until the episode ends or the agent improves enough.

Over thousands of repetitions, the agent's policy improves. It learns which actions in which states lead to the most reward in the long run.

---

## Quiz: 10 Plain-English Questions

Check your understanding. There are no tricks — every answer follows directly from the explanations above. Answers are in the collapsible sections.

---

**Q1.** You train an AI to play a video game. The AI is the ___; the game engine is the ___.

{{< collapse summary="Answer" >}}
The AI is the **agent**; the game engine is the **environment**.
{{< /collapse >}}

---

**Q2.** In a chess AI, the state is: (a) the chess board, (b) the chess pieces in the box, (c) the timer, (d) the AI's memory.

{{< collapse summary="Answer" >}}
**(a) The chess board** — specifically, the current arrangement of all pieces. The state is the description of the current situation relevant to making decisions.
{{< /collapse >}}

---

**Q3.** An agent always picks the action that worked best yesterday, never trying anything new. This is called pure ___. What is its downside?

{{< collapse summary="Answer" >}}
Pure **exploitation**. The downside: it never discovers whether a better option exists. It can get permanently stuck with a mediocre strategy.
{{< /collapse >}}

---

**Q4.** The reward for a robot reaching its goal is +10. But the robot gets a -1 penalty for every step it takes. Why add the penalty?

{{< collapse summary="Answer" >}}
Without the step penalty, the robot has no incentive to reach the goal quickly — it could wander randomly for a long time and still eventually get +10. The -1 per step **encourages the robot to take the shortest path**.
{{< /collapse >}}

---

**Q5.** Which discount factor makes an agent more short-sighted: γ = 0.99 or γ = 0.5?

{{< collapse summary="Answer" >}}
**γ = 0.5** makes the agent more short-sighted. A reward 5 steps away is worth 0.5⁵ = 0.031 of its face value — essentially ignored. With γ = 0.99, the same reward is worth 0.99⁵ ≈ 0.95 — almost full value.
{{< /collapse >}}

---

**Q6.** A recommendation system has been showing users cat videos all week and getting high clicks. On Thursday it tries a new category (travel videos) and clicks drop. Was trying travel videos exploration or exploitation?

{{< collapse summary="Answer" >}}
**Exploration** — the system tried something new to gather information, even at the cost of lower immediate reward.
{{< /collapse >}}

---

**Q7.** What is a policy?

{{< collapse summary="Answer" >}}
A **policy** is the agent's strategy — the rule that maps each state to an action (or a probability over actions). It is what the agent has learned: "In situation X, do Y."
{{< /collapse >}}

---

**Q8.** The value of a state is low even though the agent just received a high reward. How is this possible?

{{< collapse summary="Answer" >}}
Value looks **forward** — it estimates future rewards from this state onwards. A state can give a high immediate reward but be a dead end (no future rewards). For example, a chess move that captures a piece but leads to checkmate next turn has low value despite the immediate gain.
{{< /collapse >}}

---

**Q9.** An episode in a maze starts at the entrance and ends when the robot reaches the exit or exceeds 200 steps. The robot fails (runs out of steps). What was the return (undiscounted)?

{{< collapse summary="Answer" >}}
If the robot never reached the goal (reward +10) and got -1 per step for 200 steps: **return = -200** (200 × -1, no goal reward). This strongly signals that the policy was poor.
{{< /collapse >}}

---

**Q10.** Why can't an RL agent just be told the optimal policy upfront?

{{< collapse summary="Answer" >}}
Often the **optimal policy is not known in advance** — that's the point. In games like chess or Go, the space of possible states is too large to enumerate. In real-world robotics, the physics are complex and uncertain. RL discovers the policy by interacting with the environment. If we already knew the optimal policy, we wouldn't need RL.
{{< /collapse >}}

---

## Ready for More?

You now have a solid intuitive foundation. When you see the formal definitions in [Volume 1](../curriculum/volume-01/), connect them back to these analogies:

- **State** → "the snapshot"
- **Action** → "the decision"
- **Reward** → "the feedback"
- **Policy** → "the strategy"
- **Value function** → "how good is this situation for the future?"
- **Discount factor** → "how much do future rewards count?"

Next step: [Phase 1 — Math for RL](../math-for-rl/) or jump directly to [Chapter 1: The RL Framework](../curriculum/volume-01/chapter-01/).
