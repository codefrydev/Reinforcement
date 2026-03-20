---
title: "Model Evaluation"
description: "Train/test split, accuracy, precision, recall, and F1 — evaluating classifiers honestly."
date: 2026-03-20T00:00:00Z
weight: 8
draft: false
difficulty: 4
tags: ["model evaluation", "accuracy", "precision", "recall", "F1", "ml-foundations"]
keywords: ["train test split", "confusion matrix", "precision recall", "F1 score", "classifier evaluation"]
roadmap_icon: "chart"
roadmap_color: "violet"
roadmap_phase_label: "Chapter 8"
---

**Learning objectives**

- Explain why we must evaluate on held-out data, not training data.
- Construct a confusion matrix and compute TP, TN, FP, FN by hand.
- Calculate accuracy, precision, recall, and F1 from a confusion matrix.

**Concept and real-world motivation**

Imagine studying for an exam by memorizing the answer key. You would score 100% on that exact sheet — but fail any new questions. The same trap exists in ML: if you test a model on the data it trained on, you measure memorization, not learning. The fix is a **train/test split**: keep a portion of data completely separate, train the model on the rest, and evaluate only on the held-out test set.

Beyond simple accuracy, we need richer metrics. **Precision** measures "of all positive predictions, how many were actually positive?" **Recall** measures "of all actual positives, how many did we catch?" **F1** is their harmonic mean — a single number that balances both. In RL, we evaluate agents on new environments or random seeds they were never trained on — exactly the same idea of honest held-out evaluation.

**Illustration:** Metric comparison for a sample classifier.

{{< chart type="bar" palette="learning" title="Classifier Metrics" labels="Accuracy, Precision, Recall, F1" data="0.85, 0.82, 0.79, 0.80" xLabel="Metric" yLabel="Score" >}}

**Exercise:** Given `predictions = [1,0,1,1,0,1,0,0]` and `true_labels = [1,0,0,1,0,1,1,0]`:
1. Compute the confusion matrix (TP, TN, FP, FN) by hand.
2. Compute accuracy, precision, recall, and F1.
3. Verify all four metrics with `sklearn.metrics`.

{{< pyrepl code="import numpy as np\n\npredictions = [1, 0, 1, 1, 0, 1, 0, 0]\ntrue_labels  = [1, 0, 0, 1, 0, 1, 1, 0]\n\n# TODO: compute TP, TN, FP, FN\n# TP = predicted 1, actually 1\n# TN = predicted 0, actually 0\n# FP = predicted 1, actually 0\n# FN = predicted 0, actually 1\nTP = sum(p == 1 and t == 1 for p, t in zip(predictions, true_labels))\nTN = None  # TODO\nFP = None  # TODO\nFN = None  # TODO\n\n# TODO: compute metrics\naccuracy  = None  # (TP + TN) / total\nprecision = None  # TP / (TP + FP)\nrecall    = None  # TP / (TP + FN)\nf1        = None  # 2 * P * R / (P + R)\n\nprint(f'TP={TP}, TN={TN}, FP={FP}, FN={FN}')\nprint(f'Accuracy={accuracy:.3f}, Precision={precision:.3f}')\nprint(f'Recall={recall:.3f}, F1={f1:.3f}')\n# expected: Accuracy≈0.75, Precision≈0.80, Recall≈0.75, F1≈0.77" height="280" >}}

**Professor's hints**

- Loop over `zip(predictions, true_labels)` and accumulate four counters, one per confusion matrix cell.
- Precision's denominator is everything you predicted as positive (TP + FP). Recall's denominator is everything that was actually positive (TP + FN).
- F1 uses the harmonic mean: \\(F_1 = \frac{2PR}{P + R}\\). Harmonic means penalize extreme imbalances between P and R more than the arithmetic mean would.

**Common pitfalls**

- **Testing on training data:** Never call `score()` or compute metrics on your training set to claim model quality. Always use held-out data.
- **Swapping FP and FN:** FP = you said positive, it was negative. FN = you said negative, it was positive. Get the denominators of precision and recall right.
- **Using accuracy alone on imbalanced data:** If 95% of samples are class 0, a model that always predicts 0 gets 95% accuracy. Precision and recall reveal the truth.

{{< collapse summary="Worked solution" >}}
```python
predictions = [1, 0, 1, 1, 0, 1, 0, 0]
true_labels  = [1, 0, 0, 1, 0, 1, 1, 0]

TP = sum(p == 1 and t == 1 for p, t in zip(predictions, true_labels))  # 3
TN = sum(p == 0 and t == 0 for p, t in zip(predictions, true_labels))  # 3
FP = sum(p == 1 and t == 0 for p, t in zip(predictions, true_labels))  # 1
FN = sum(p == 0 and t == 1 for p, t in zip(predictions, true_labels))  # 1

accuracy  = (TP + TN) / len(predictions)   # 0.75
precision = TP / (TP + FP)                 # 3/4 = 0.75... wait TP=3, FP=1 → 0.75
recall    = TP / (TP + FN)                 # 3/4 = 0.75
f1        = 2 * precision * recall / (precision + recall)  # 0.75

# sklearn verification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(accuracy_score(true_labels, predictions))
print(precision_score(true_labels, predictions))
print(recall_score(true_labels, predictions))
print(f1_score(true_labels, predictions))
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For 5 predictions `[1,0,1,0,1]` and true labels `[1,1,1,0,0]`, compute accuracy by hand. Count TP, TN, FP, FN, then divide.

{{< pyrepl code="preds  = [1, 0, 1, 0, 1]\ntruths = [1, 1, 1, 0, 0]\n# TODO: compute accuracy without any library\n# expected: accuracy = 3/5 = 0.60\naccuracy = None\nprint(f'Accuracy: {accuracy}')" height="180" >}}

2. **Coding:** Write a function `confusion_counts(y_true, y_pred)` that returns a dict `{'TP': ..., 'TN': ..., 'FP': ..., 'FN': ...}` for any binary classification lists.
3. **Challenge:** Generate a random binary classifier (`np.random.randint(0, 2, 100)`) and a random true label array. Compute all four metrics. Then try a classifier that always predicts 1 — compare precision and recall.
4. **Variant:** Change `predictions[6]` from 0 to 1. Recompute precision and recall. Which one changes? Why?
5. **Debug:** The code below has a bug — FP and FN are swapped in the precision and recall formulas. Find and fix it.

{{< pyrepl code="TP, TN, FP, FN = 3, 3, 1, 1\n# BUG: FP and FN are swapped below\nprecision = TP / (TP + FN)   # BUG: should be TP + FP\nrecall    = TP / (TP + FP)   # BUG: should be TP + FN\nf1 = 2 * precision * recall / (precision + recall)\nprint(f'precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}')\n# TODO: fix both lines\n# expected: precision=0.75, recall=0.75, f1=0.75" height="200" >}}

6. **Conceptual:** Give a real-world example where high recall matters more than high precision (e.g. cancer screening). Now give one where high precision matters more.
7. **Recall:** From memory, write the formulas for precision, recall, and F1 in terms of TP, TN, FP, FN.
