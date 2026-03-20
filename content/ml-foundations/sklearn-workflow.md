---
title: "Scikit-Learn Workflow"
description: "The full sklearn pipeline: fit, predict, score, and comparing multiple models."
date: 2026-03-20T00:00:00Z
weight: 13
draft: false
difficulty: 4
tags: ["scikit-learn", "sklearn", "pipeline", "model comparison", "ml-foundations"]
keywords: ["scikit-learn workflow", "fit predict score", "sklearn pipeline", "model comparison", "StandardScaler"]
roadmap_icon: "terminal"
roadmap_color: "purple"
roadmap_phase_label: "Chapter 13"
---

**Learning objectives**

- Use the sklearn API (`fit` / `predict` / `score`) consistently across different model classes.
- Build a `Pipeline` that chains preprocessing and a classifier.
- Compare multiple models on the same dataset using test-set accuracy.

**Concept and real-world motivation**

Scikit-learn provides a unified API: every model has `fit(X_train, y_train)`, `predict(X_test)`, and `score(X_test, y_test)`. This consistency lets you swap models with one line of code. **Pipelines** extend this: they chain preprocessing steps (like `StandardScaler`) and a final estimator into a single object that can be fit, predicted, and cross-validated as a unit — preventing data leakage automatically.

In RL, we follow an analogous consistent pattern: **initialize** the agent → **interact** with the environment → **update** the policy → **evaluate** on new episodes. Just as sklearn pipelines chain steps, RL training loops chain environment steps, replay updates, and evaluation episodes. Having a consistent API makes experimentation fast.

**Illustration:** The sklearn pipeline flow.

{{< mermaid >}}
flowchart LR
    A["Raw Data"] --> B["Preprocessing\n(StandardScaler)"]
    B --> C["Model.fit()"]
    C --> D["Model.predict()"]
    D --> E["Evaluate\n(accuracy, F1)"]
    E --> F["Report"]
{{< /mermaid >}}

**Exercise:** Run the full sklearn workflow on the Iris dataset: load, split, train two models, and compare.

{{< pyrepl code="from sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score\n\n# Step 1: Load data\niris = load_iris()\nX, y = iris.data, iris.target\n\n# Step 2: Train/test split 80/20\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42)\n\n# Step 3: Fit LogisticRegression\nlr = LogisticRegression(max_iter=200)\n# TODO: fit lr on training data\n# TODO: predict on test data\n# TODO: compute accuracy\nlr_acc = None\n\n# Step 4: Fit DecisionTreeClassifier\ndt = DecisionTreeClassifier(random_state=42)\n# TODO: fit, predict, accuracy\ndt_acc = None\n\nprint(f'Logistic Regression accuracy: {lr_acc:.3f}')\nprint(f'Decision Tree accuracy:       {dt_acc:.3f}')\n# expected: both > 0.90 on Iris" height="320" >}}

**Professor's hints**

- The sklearn pattern is always: `model.fit(X_train, y_train)`, then `y_pred = model.predict(X_test)`, then `accuracy_score(y_test, y_pred)`. Or shortcut: `model.score(X_test, y_test)`.
- `LogisticRegression(max_iter=200)` — increase `max_iter` if you see a ConvergenceWarning.
- `DecisionTreeClassifier(random_state=42)` sets the random seed for reproducibility. Without it, results may vary across runs.

**Common pitfalls**

- **Calling `fit` on test data:** Never `model.fit(X_test, y_test)`. The model must only see `X_train` during fitting. Evaluation uses a completely separate `X_test`.
- **Forgetting to scale for logistic regression:** Logistic regression converges faster and more reliably when features are on the same scale. Use `StandardScaler` in a Pipeline.
- **Using `score` on training data to report model quality:** `model.score(X_train, y_train)` is training accuracy — it measures memorization. Always report `model.score(X_test, y_test)`.

{{< collapse summary="Worked solution" >}}
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))

print(f'LR: {lr_acc:.3f}, DT: {dt_acc:.3f}')
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Use `Pipeline` with `StandardScaler` and `LogisticRegression` on Iris. Confirm you get the same or better accuracy than the unscaled version above.

{{< pyrepl code="from sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\niris = load_iris()\nX, y = iris.data, iris.target\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# TODO: build a Pipeline with StandardScaler + LogisticRegression\npipe = None  # Pipeline([('scaler', ...), ('lr', ...)])\n# TODO: fit the pipeline on training data\n# TODO: score on test data\nprint(f'Pipeline accuracy: {pipe.score(X_test, y_test):.3f}')\n# expected: ≥ 0.93" height="220" >}}

2. **Coding:** Run `cross_val_score` (5-fold) on both `LogisticRegression` and `DecisionTreeClassifier` using a `Pipeline` with `StandardScaler`. Report mean ± std for each.
3. **Challenge:** Compare 4 classifiers on Iris: `LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier(n_neighbors=5)`, and `RandomForestClassifier(n_estimators=100)`. Print a table of train accuracy and test accuracy for each. Which overfits the most?
4. **Variant:** Try `DecisionTreeClassifier(max_depth=1)`, `max_depth=3`, and `max_depth=None` (unlimited). Plot test accuracy vs max_depth. At what depth does overfitting start?
5. **Debug:** The code below has a bug — `fit` is called on the test set instead of the training set. Find and fix it.

{{< pyrepl code="from sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\n\niris = load_iris()\nX, y = iris.data, iris.target\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nlr = LogisticRegression(max_iter=200)\n# BUG: fitting on test data instead of train data\nlr.fit(X_test, y_test)\nacc = lr.score(X_test, y_test)\nprint(f'Accuracy: {acc:.3f}')  # deceptively high\n# TODO: fix so model trains on X_train, y_train\n# expected: should see fair accuracy ~0.93 on separate test set" height="200" >}}

6. **Conceptual:** Why does a `Pipeline` prevent data leakage during cross-validation, whereas fitting a `StandardScaler` separately on all data before splitting does not?
7. **Recall:** From memory, write the three core sklearn API calls for a classification workflow, and explain what each does.
