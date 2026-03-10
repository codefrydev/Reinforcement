---
title: "Chapter 90: Communication in MARL"
description: "Agents output message + action; train for coordination task."
date: 2026-03-10T00:00:00Z
weight: 90
draft: false
tags: ["MARL", "communication", "coordination", "message", "curriculum"]
keywords: ["communication in MARL", "coordination", "message and action", "multi-agent"]
---

**Learning objectives**

- **Implement** a simple **communication protocol**: each agent outputs a **message** (e.g. a vector) in addition to its **action**; the message is fed into other agents' policies (e.g. as part of their observation at the next step).
- **Train** agents to solve a task that **requires coordination** (e.g. two agents must swap positions or colors, or meet at a target) using this communication.
- **Compare** with the same task **without** communication (each agent sees only local observation) and report improvement in return or success rate.
- **Explain** how learned communication can encode information (e.g. "I am going left") that helps coordination.
- **Relate** communication in MARL to **dialogue** (multi-turn interaction) and **robot navigation** (multi-robot signaling).

**Concept and real-world RL**

**Communication** in multi-agent RL allows agents to send **messages** (discrete or continuous) that other agents observe. The message is often produced by the same policy that produces the action (e.g. π(a, m | o, m_prev) or a separate message head). Agents are trained end-to-end so that useful communication emerges (e.g. to signal intent or share information). Tasks that **require coordination** (e.g. swap colors, meet at a location, divide roles) benefit from communication when the local observation is insufficient. In **dialogue** and **robot navigation**, explicit communication (or learned signaling) is a natural extension of MARL.

**Where you see this in practice:** CommNet, TarMAC, and learned communication in MARL; multi-robot coordination; emergent language.

**Illustration (communication):** Agents that can send messages often achieve higher return on coordination tasks. The chart below compares return with vs without communication (e.g. swap-colors task).

{{< chart type="bar" title="Return (coordination task)" labels="No comm, With comm" data="40, 95" >}}

**Exercise:** Implement a simple communication protocol where agents output a message alongside their action. The message is fed into other agents' policies. Train them to solve a task that requires coordination (e.g., "two agents need to swap colors").

**Professor's hints**

- **Message:** Each agent i outputs (a_i, m_i). m_i can be a fixed-size vector (e.g. 4 dims). At the next step, agent j's observation includes the messages from others: o_j' = (o_j, m_1,...,m_n) or (o_j, m_{-j}). So the policy is π_i(a_i, m_i | o_i, m_others).
- **Swap colors task:** Two agents; each has a color (e.g. red, blue). They must swap positions (or swap colors). Without communication, they may not know the other's intent; with messages they can signal "I go left" or "meet at center." Define a small grid or graph and reward for successful swap within T steps.
- **Training:** Use PPO or Q-learning; the message is part of the policy output. Backprop through the message into the policy. Messages can be continuous (e.g. tanh) or discrete (e.g. one-hot, then use Gumbel-softmax or straight-through).
- **Baseline:** Same task, same architecture, but message is zero or not used (or not fed to others). Compare success rate or return.

**Common pitfalls**

- **Message not used:** Ensure the other agents' policies actually receive and use the message (e.g. concatenate to observation). Otherwise communication has no effect.
- **Credit assignment:** The reward is often shared (team reward); the agent that sent a useful message may not get direct credit. Training with team return usually suffices for coordination.
- **Task too simple:** If the task can be solved without communication (e.g. by luck or simple policy), the benefit of communication may be small. Choose a task where coordination is clearly needed (e.g. swap requires both to move in a coordinated way).

{{< collapse summary="Worked solution (warm-up: communication in MARL)" >}}
**Key idea:** Agents can send messages (discrete or continuous) that other agents observe. We train the full system (policies + message interpretation) so that the messages help coordination. The message can be part of the observation for the receiver; the sender’s policy outputs (action, message). Tasks like "swap positions" or "meet at a location" need coordination; without communication, independent policies may fail. CommNet and TarMAC are examples.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In the "swap colors" task, what information might one agent need from the other that is not in its local observation?
2. **Coding:** Implement 2 agents on a 3×3 grid; each has a color (red/blue). Goal: swap positions (so red is where blue was and vice versa). Add a 2-dimensional message per agent, broadcast to the other. Train with PPO and team reward. Compare success rate over 500 episodes with and without communication (zero message or message not passed).
3. **Challenge:** Use **discrete** messages (e.g. 4 symbols: "left," "right," "up," "down"). Train with Gumbel-softmax or REINFORCE for the message. Do agents learn interpretable symbols? Visualize message usage in a few episodes.
