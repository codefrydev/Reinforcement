---
title: "Cross-Validation and Overfitting"
description: "K-fold cross-validation, overfitting vs underfitting, and the bias-variance tradeoff."
date: 2026-03-20T00:00:00Z
weight: 9
draft: false
difficulty: 4
tags: ["cross-validation", "overfitting", "underfitting", "bias-variance", "ml-foundations"]
keywords: ["k-fold cross-validation", "overfitting", "underfitting", "bias variance tradeoff", "model selection"]
roadmap_icon: "sparkles"
roadmap_color: "blue"
roadmap_phase_label: "Chapter 9"
---

**Learning objectives**

- Explain why a single train/test split gives a noisy estimate of model quality.
- Implement K-fold cross-validation from scratch using array slicing.
- Describe the bias-variance tradeoff and identify overfitting and underfitting from error curves.

**Concept and real-world motivation**

A single train/test split is like grading a student on one exam. If that exam happens to cover topics they know well (or badly), the grade is misleading. **K-fold cross-validation** fixes this: divide data into K equal parts ("folds"), then train on K-1 folds and test on the remaining fold, rotating K times. Average the K test scores for a much more stable estimate.

The deeper issue K-fold guards against is **overfitting**: a model that memorizes training data but fails on new data. The opposite — a model too simple to capture the pattern — is **underfitting**. The **bias-variance tradeoff** names this tension: simple models have high bias (systematic error), complex models have high variance (sensitivity to training data). The sweet spot minimizes total error on held-out data. In RL, we evaluate agents across multiple random seeds — exactly the K-fold idea applied to policy evaluation.

**Illustration:** Test error shows a U-shape as model complexity grows — low at the sweet spot.

{{< chart type="line" palette="comparison" title="Error vs Model Complexity" labels="1,2,3,4,5,6" data="0.85,0.82,0.80,0.79,0.82,0.88" xLabel="Complexity" yLabel="Test Error" >}}

**Exercise:** Implement 5-fold cross-validation manually on a synthetic dataset, without using sklearn's `cross_val_score`.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n\n# 20 samples, 2 features, binary labels\nX = np.random.randn(20, 2)\ny = (X[:, 0] + X[:, 1] > 0).astype(int)\n\nK = 5\nfold_size = len(X) // K\nscores = []\n\nfor fold in range(K):\n    # TODO: split into train and test using array slicing\n    # test indices: fold*fold_size to (fold+1)*fold_size\n    test_idx  = list(range(fold * fold_size, (fold + 1) * fold_size))\n    train_idx = None  # TODO: all indices NOT in test_idx\n\n    X_train, X_test = X[train_idx], X[test_idx]\n    y_train, y_test = y[train_idx], y[test_idx]\n\n    # simple classifier: predict majority class from training set\n    majority = int(np.mean(y_train) >= 0.5)\n    preds = np.full(len(y_test), majority)\n    acc = np.mean(preds == y_test)\n    scores.append(acc)\n\nprint(f'Fold scores: {[round(s,3) for s in scores]}')\nprint(f'Mean CV score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')\n# expected: mean around 0.50-0.65 for majority classifier" height="300" >}}

**Professor's hints**

- `train_idx` = all indices in `range(20)` that are NOT in `test_idx`. Use a list comprehension: `[i for i in range(len(X)) if i not in test_idx]`.
- You can use any classifier here. A majority-vote baseline is fine for the exercise structure. Try replacing it with a real model from sklearn if you have it available.
- The variance of CV scores tells you how stable the estimate is. High variance across folds means you need more data or a simpler model.

**Common pitfalls**

- **Overlapping folds:** Ensure test indices are non-overlapping across iterations. Off-by-one errors (`range(fold * fold_size, (fold+1) * fold_size)` vs. `+1`) cause overlap and data leakage.
- **Leaking test data into preprocessing:** If you normalize with statistics computed on all data before splitting, information from the test fold contaminates training. Always fit scalers only on the training fold.
- **Using K=len(data) (leave-one-out) blindly:** LOOCV is unbiased but has high variance and is slow on large datasets. K=5 or K=10 is usually the right default.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np
np.random.seed(42)
X = np.random.randn(20, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

K = 5
fold_size = len(X) // K
scores = []

for fold in range(K):
    test_idx  = list(range(fold * fold_size, (fold + 1) * fold_size))
    train_idx = [i for i in range(len(X)) if i not in test_idx]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    majority = int(np.mean(y_train) >= 0.5)
    preds = np.full(len(y_test), majority)
    acc = np.mean(preds == y_test)
    scores.append(acc)

print(f'Mean CV: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For a dataset of 10 samples, list the exact train and test indices for each fold when K=5 (fold_size=2). Write them out as lists.

{{< pyrepl code="K = 5\nn = 10\nfold_size = n // K\nfor fold in range(K):\n    test_idx  = list(range(fold * fold_size, (fold + 1) * fold_size))\n    train_idx = [i for i in range(n) if i not in test_idx]\n    print(f'Fold {fold}: train={train_idx}, test={test_idx}')\n# TODO: verify no overlap between train and test in each fold" height="180" >}}

2. **Coding:** Compare 5-fold CV scores for a degree-1 polynomial (underfitting) and a degree-5 polynomial (overfitting) on the same 20-sample dataset. Which has lower test error?

{{< pyrepl code="import numpy as np\nnp.random.seed(0)\nx = np.linspace(0, 1, 20)\ny = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(20)\n\n# TODO: fit degree-1 and degree-5 polynomial, compare train vs test error\n# Hint: np.polyfit(x_train, y_train, deg) and np.polyval(coeffs, x_test)\ndeg1_train_err = None\ndeg5_train_err = None\ndeg1_test_err  = None\ndeg5_test_err  = None\n\nprint(f'Degree-1  train={deg1_train_err:.3f}  test={deg1_test_err:.3f}')\nprint(f'Degree-5  train={deg5_train_err:.3f}  test={deg5_test_err:.3f}')\n# expected: degree-5 has lower train error but higher test error (overfitting)" height="220" >}}

3. **Challenge:** Implement stratified K-fold from scratch: ensure each fold has approximately the same class balance as the full dataset. Compare CV scores with and without stratification on a heavily imbalanced dataset (90% class 0, 10% class 1).
4. **Variant:** Re-run the main exercise with K=3 and K=10. How does mean CV score and its standard deviation change? Why is K=10 called "leave-more-out" when fold sizes are larger?
5. **Debug:** The code below has a bug — fold indices overlap because the range end is wrong. Find and fix it.

{{< pyrepl code="import numpy as np\nX = np.arange(10)\nK = 5\nfold_size = len(X) // K\n\nfor fold in range(K):\n    # BUG: end index is off by one — folds overlap\n    test_idx = list(range(fold * fold_size, fold * fold_size + fold_size + 1))\n    train_idx = [i for i in range(len(X)) if i not in test_idx]\n    print(f'fold {fold}: test={test_idx}, len={len(test_idx)}')\n# TODO: fix the test_idx range so each fold has exactly fold_size elements\n# expected: each fold has exactly 2 elements, no overlap" height="200" >}}

6. **Conceptual:** Why is the harmonic mean (used in F1) more appropriate than the arithmetic mean when averaging precision and recall? Does the same logic apply to averaging CV fold scores?
7. **Recall:** From memory, describe the K-fold algorithm in 4 steps. Then write the formula for the CV score as the average of fold test scores.
