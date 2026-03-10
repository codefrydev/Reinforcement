---
title: "Chapter 79: Offline-to-Online Finetuning"
description: "Pretrain SAC offline; finetune online; Q-filter for bad actions."
date: 2026-03-10T00:00:00Z
weight: 79
draft: false
tags: ["offline-to-online", "SAC", "Q-filter", "finetuning", "curriculum"]
keywords: ["offline-to-online", "finetune", "SAC", "Q-filter", "pretrain offline"]
---

**Learning objectives**

- **Pretrain** an SAC (or similar) agent **offline** on a fixed dataset (e.g. from a mix of policies or from Chapter 71).
- **Finetune** the agent **online** by continuing training with environment interaction.
- **Compare** the learning curve (return vs steps) of finetuning from offline pretraining vs training from scratch.
- **Implement** a **Q-filter**: when updating the policy, avoid or downweight updates that use actions for which Q is below a threshold (to avoid reinforcing "bad" actions that could destabilize the policy).
- **Relate** offline-to-online to **recommendation** (pretrain on logs, then A/B test) and **healthcare** (pretrain on historical data, then cautious online updates).

**Concept and real-world RL**

**Offline-to-online finetuning** first trains a policy **offline** on a fixed dataset (no environment interaction), then **continues training online** with interaction. This is useful when we have a lot of logged data but also the ability to collect new data. The risk is that the offline policy may have learned to avoid OOD actions (e.g. with CQL), and when we go online, the policy might be too conservative or the value estimates might be off. A **Q-filter** helps: when taking a policy gradient step (or updating the actor), only use or upweight transitions where Q(s, a) is above a threshold (e.g. the dataset average), so we do not reinforce actions the critic thinks are bad. In **recommendation** and **healthcare**, we often pretrain on historical data and then carefully finetune with limited online feedback.

**Where you see this in practice:** Offline pretraining + online finetuning; Q-filter and similar safeguards; safe deployment from batch data.

**Illustration (offline pretrain + online):** Pretraining on offline data then fine-tuning online often reaches good return faster than training from scratch. The chart below compares learning curves.

{{< chart type="line" title="Return: from scratch vs pretrain+online" labels="0, 50k, 100k, 150k, 200k" data="0, 500, 1500, 3000, 4500" >}}

**Exercise:** Pretrain an SAC agent offline using a large dataset. Then continue training online in the environment. Compare the learning curve with training from scratch. Use Q-filter to avoid updating with bad actions.

**Professor's hints**

- **Offline phase:** Use the same setup as Chapter 71 or 72: collect or load a dataset (e.g. 500k transitions from a random or mixed policy on Hopper). Train SAC (or CQL-SAC) offline until convergence. Save the policy and critic.
- **Online phase:** Load the pretrained networks. Run the environment; add new transitions to the replay buffer (you can mix with offline data or gradually replace). Continue SAC updates. Plot return vs total steps (including offline steps if you count them).
- **Q-filter:** When computing the actor loss, for each (s, a) in the batch, if Q(s, a) < threshold (e.g. 25th percentile of Q on the batch, or a fixed value), set the weight for that sample to 0 (or a small value). So we do not encourage the policy to take actions the critic thinks are bad. Implement as: mask = (Q(s, a) >= threshold).float(); actor_loss = (mask * ...).sum() / mask.sum().
- **Comparison:** Run two curves—(1) train from scratch online (no offline), (2) offline pretrain then online finetune. Plot both. Optionally add (3) offline + online with Q-filter.

**Common pitfalls**

- **Forgetting offline data:** If you only add online data to the buffer and the buffer is small, you may forget the offline distribution; consider keeping a fraction of offline data in the buffer or using a larger buffer.
- **Q-filter too aggressive:** If the threshold is too high, you filter out most updates and learning is slow. Use a percentile (e.g. 25th) so that a fraction of transitions are always used.
- **Evaluation during online:** Evaluate the policy periodically (e.g. 5 episodes every 5k steps) without exploration noise to get a fair learning curve.

{{< collapse summary="Worked solution (warm-up: offline-to-online)" >}}
**Key idea:** We pretrain on offline data (e.g. CQL or BC), then fine-tune with online interaction. The pretrained policy is a good initialization and stays near the data distribution initially; we can use a small bonus for in-distribution actions or a conservative Q so that early online updates do not destroy the policy. Then we gradually allow more exploration. This combines the sample efficiency of offline with the improvement of online.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why might we want to "filter" policy updates so that we do not reinforce actions with low Q-values when finetuning from offline to online?
2. **Coding:** Pretrain SAC offline on 200k Hopper transitions. Then finetune online for 100k steps. Plot return every 5k steps. Compare with SAC from scratch for 100k steps. Which reaches higher return faster?
3. **Challenge:** Implement the Q-filter: threshold = 25th percentile of Q(s, a) on the current batch. Compare online finetuning with and without Q-filter (same offline pretraining). Does Q-filter reduce instability or improve final return?
