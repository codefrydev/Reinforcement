---
title: "What is Machine Learning?"
description: "Three types of ML: supervised, unsupervised, and reinforcement — and why learning from data beats hand-written rules."
date: 2026-03-20T00:00:00Z
weight: 1
draft: false
difficulty: 4
tags: ["machine learning", "supervised learning", "unsupervised learning", "reinforcement learning", "ml-foundations"]
keywords: ["what is machine learning", "supervised vs unsupervised", "types of machine learning", "ML for RL", "learning from data"]
roadmap_icon: "book"
roadmap_color: "blue"
roadmap_phase_label: "Chapter 1"
---

**Learning objectives**

- Distinguish traditional rule-based programming from machine learning.
- Name and describe the three main types of ML: supervised, unsupervised, and reinforcement learning.
- Classify a new problem as supervised, unsupervised, or reinforcement learning given its description.

**Concept and real-world motivation**

In traditional programming, a human writes explicit rules: `if email contains "free money" → spam`. This works for simple cases but breaks the moment the world gets complicated. A spam filter based on hand-written rules fails against new tricks; a chess program based on if/else trees cannot compete with millions of possible board positions. **Machine learning** takes a different approach: instead of programming rules, we show the machine *examples* and let it figure out the patterns.

The three types of ML differ in what kind of signal the machine learns from. In **supervised learning**, every example comes with a correct answer (label) — like a dataset of emails labeled "spam" or "not spam." In **unsupervised learning**, there are no labels — the machine finds hidden structure on its own, like grouping customers by purchase behavior. In **reinforcement learning**, there are no labels at all, only a reward signal: the agent tries things in an environment and learns which actions lead to more reward. This is exactly how we will train RL agents in the rest of this course — **RL is a third type of ML**, and everything you learn about supervised learning here will reappear inside RL algorithms.

**Illustration:** The bar chart below shows the rough proportion of real-world ML problem types encountered in industry (supervised problems are by far the most common, which is why we spend the most time on them).

{{< chart type="bar" palette="comparison" title="ML problem types in practice" labels="Supervised, Unsupervised, Reinforcement" data="1000, 300, 100" xLabel="Problem type" yLabel="Approximate count" >}}

**Exercise:** Eight problems are listed below. For each one, classify it as `supervised`, `unsupervised`, or `reinforcement`. Fill in the `my_answers` list and run the cell to check your work.

{{< pyrepl code="# Classify each problem. Fill in your answers:\n# 1. Predict house price from size and location\n# 2. Group customers by purchase behavior\n# 3. Train a robot to walk on uneven terrain\n# 4. Classify email as spam or not spam\n# 5. Find unusual transactions in a bank (no labels)\n# 6. Learn to play chess by playing games\n# 7. Predict tomorrow's temperature from weather history\n# 8. Cluster news articles into topics\n\nmy_answers = [\n    # TODO: fill in each answer as 'supervised', 'unsupervised', or 'reinforcement'\n    '',  # 1. house price\n    '',  # 2. customer groups\n    '',  # 3. robot walk\n    '',  # 4. spam\n    '',  # 5. anomaly detection\n    '',  # 6. chess\n    '',  # 7. weather\n    '',  # 8. news clusters\n]\n\ncorrect = ['supervised','unsupervised','reinforcement','supervised','unsupervised','reinforcement','supervised','unsupervised']\nfor i, (a, c) in enumerate(zip(my_answers, correct), 1):\n    mark = 'OK' if a == c else f'WRONG (correct: {c})'\n    print(f'Problem {i}: {a or \"(blank)\"} -> {mark}')" height="300" >}}

**Professor's hints**

- Ask yourself: "Is there a correct answer (label) for each example?" If yes → supervised.
- Ask yourself: "Is an agent taking actions and receiving rewards?" If yes → reinforcement learning.
- If neither — the algorithm is finding structure without guidance → unsupervised.
- Anomaly detection (problem 5) has no "here is the anomaly" label in the training data — the model learns what "normal" looks like and flags departures.

**Common pitfalls**

- **Confusing RL with supervised learning:** In RL, the agent does not receive the "correct action" for each state — it only receives a reward after a sequence of actions. This is a fundamentally different signal from a labeled dataset.
- **Thinking unsupervised = no learning:** Unsupervised methods learn rich structure (clusters, dimensions, densities) — they just do so without human-provided labels.
- **Assuming RL requires a game:** RL applies to any sequential decision-making problem: robotics, recommendation systems, resource scheduling, and more.

{{< collapse summary="Worked solution" >}}
Here are the correct classifications and the reasoning:

1. **House price prediction** → `supervised` — Each house has a known sale price (label). The model learns the mapping features → price.
2. **Customer grouping** → `unsupervised` — No pre-labeled clusters exist. K-means or similar algorithms find groups from the data itself.
3. **Robot walking** → `reinforcement` — The robot receives reward for staying upright and penalized for falling. No labeled "correct joint angles" exist.
4. **Spam classification** → `supervised` — Emails are labeled spam/not-spam. The model learns from those labels.
5. **Bank anomaly detection** → `unsupervised` — Normal transactions are not individually labeled. The model learns the distribution of normal and flags deviations.
6. **Chess** → `reinforcement` — The agent plays games and receives +1 for winning, 0 for draw, -1 for losing. It learns by trial and error.
7. **Weather prediction** → `supervised` — Historical records pair (today's features) → (tomorrow's temperature). Each training example has a label.
8. **News clustering** → `unsupervised` — Articles are not pre-categorized. Topic models or clustering algorithms find the groups.

```python
correct = ['supervised','unsupervised','reinforcement',
           'supervised','unsupervised','reinforcement',
           'supervised','unsupervised']
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Write a simple if/else "classifier" that predicts spam based on keywords. Then try it on a new email it gets wrong. This shows the limit of rule-based approaches.

{{< pyrepl code="def rule_based_spam(email_text):\n    # TODO: return True if email_text looks like spam, False otherwise\n    # Use simple string checks like 'free' in email_text\n    pass\n\ntest_emails = [\n    ('Win a FREE iPhone now!!!', True),\n    ('Meeting at 3pm tomorrow', False),\n    ('Claim your prize — limited time', True),\n    ('Can you review my pull request?', False),\n    ('You have been selected for exclusive offer', True),\n]\nfor text, label in test_emails:\n    pred = rule_based_spam(text)\n    mark = 'OK' if pred == label else 'MISS'\n    print(f'{mark}: \"{text[:40]}...\" -> pred={pred}, actual={label}')" height="240" >}}

2. **Coding:** Add two more test emails to the list above that your rule-based classifier gets wrong. This demonstrates why data-driven ML is needed.
3. **Challenge:** A recommendation system suggests movies based on watch history. Is it supervised, unsupervised, or reinforcement learning? Argue for more than one interpretation — under what framing is it each type?
4. **Variant:** Suppose you have a dataset of customer transactions and you know which transactions were fraudulent (labeled by a human review team). How does this change anomaly detection from unsupervised to supervised? What are the pros and cons of each approach?
5. **Debug:** The code below tries to check answers but has a bug — it always prints "OK" even for wrong answers. Find and fix it.

{{< pyrepl code="my_answers = ['supervised', 'supervised', 'supervised']  # all wrong\ncorrect    = ['supervised', 'unsupervised', 'reinforcement']\n\nfor i, (a, c) in enumerate(zip(my_answers, correct), 1):\n    # BUG: the comparison is wrong\n    if a != c:\n        mark = 'OK'\n    else:\n        mark = f'WRONG (correct: {c})'\n    print(f'Problem {i}: {mark}')\n\n# TODO: fix the if/else logic so OK means correct and WRONG means incorrect" height="200" >}}

6. **Conceptual:** Explain in one paragraph why reinforcement learning is harder than supervised learning. What makes the reward signal a less informative teaching signal than a labeled dataset?
7. **Recall:** Name the three types of machine learning and give one real-world example of each from memory. Write your answer before looking at the page.
