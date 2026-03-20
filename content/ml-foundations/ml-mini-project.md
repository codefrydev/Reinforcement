---
title: "ML Mini-Project: Wine Classification"
description: "End-to-end ML project combining loading, exploration, preprocessing, training, and evaluation."
date: 2026-03-20T00:00:00Z
weight: 14
draft: false
difficulty: 4
tags: ["mini-project", "wine dataset", "classification", "end-to-end", "ml-foundations"]
keywords: ["end-to-end machine learning", "wine dataset", "sklearn pipeline", "model comparison", "classification project"]
roadmap_icon: "rocket"
roadmap_color: "rose"
roadmap_phase_label: "Chapter 14 · Mini-Project"
---

**Learning objectives**

- Execute a complete ML workflow from raw data to model comparison.
- Apply `StandardScaler`, train multiple classifiers, and evaluate with accuracy, precision, and recall.
- Interpret results and make a justified model choice.

**Concept and real-world motivation**

This page is a mini-project that integrates every concept from the ML Foundations section. There is no new theory — only application. Real ML work looks exactly like this: load data, explore it, preprocess, train several models, evaluate honestly on held-out data, and compare results systematically.

The same workflow applies to RL evaluation: load or generate trajectories, preprocess states, train a value function or policy, evaluate on unseen episodes, and compare agent variants. The "best model" in supervised learning is the one with the best test metrics; the "best agent" in RL is the one that maximizes expected return across new environments. This project is your bridge between the two worlds.

**Illustration:** Compare accuracy across three classifiers.

{{< chart type="bar" palette="comparison" title="Model Accuracy on Wine Test Set" labels="Logistic Reg,Decision Tree,KNN (k=5)" data="0.97,0.94,0.96" xLabel="Model" yLabel="Test Accuracy" >}}

**Exercise — Full pipeline on the Wine dataset (Steps 1–4):**

Load and explore the Wine dataset, preprocess, and train three models.

{{< pyrepl code="from sklearn.datasets import load_wine\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score\nimport numpy as np\n\n# Step 1: Load data\nwine = load_wine()\nX, y = wine.data, wine.target\nprint(f'Shape: {X.shape}, Classes: {np.unique(y)}')\nprint(f'Feature names: {wine.feature_names[:4]}...')\n\n# Step 2: Train/test split 80/20\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y)\n\n# Step 3: Preprocess — StandardScaler fit on train only\nscaler = StandardScaler()\n# TODO: fit scaler on X_train, then transform both X_train and X_test\nX_train_s = None  # scaler.fit_transform(X_train)\nX_test_s  = None  # scaler.transform(X_test)\n\n# Step 4: Train 3 models\nmodels = {\n    'LogisticRegression': LogisticRegression(max_iter=1000),\n    'DecisionTree':       DecisionTreeClassifier(random_state=42),\n    'KNN_k5':             KNeighborsClassifier(n_neighbors=5),\n}\nresults = {}\nfor name, model in models.items():\n    # TODO: fit, predict, compute accuracy precision recall\n    model.fit(X_train_s, y_train)\n    y_pred = model.predict(X_test_s)\n    results[name] = {\n        'accuracy':  round(accuracy_score(y_test, y_pred), 3),\n        'precision': round(precision_score(y_test, y_pred, average='macro'), 3),\n        'recall':    round(recall_score(y_test, y_pred, average='macro'), 3),\n    }\n\nfor name, m in results.items():\n    print(f'{name:20s}: acc={m[\"accuracy\"]}, prec={m[\"precision\"]}, rec={m[\"recall\"]}')\n# expected: LogReg typically 0.97+, DTree ~0.92-0.97, KNN ~0.94-0.97" height="400" >}}

**Professor's hints**

- `scaler.fit_transform(X_train)` fits AND transforms in one step. Then `scaler.transform(X_test)` applies the SAME scaling (do not refit on test — that would be data leakage).
- `precision_score(..., average='macro')` averages precision across all 3 classes equally. Use `'weighted'` if classes are imbalanced.
- `stratify=y` in `train_test_split` ensures all 3 wine classes appear in both train and test in the right proportions.

**Common pitfalls**

- **Data leakage via scaler:** `scaler.fit_transform(X)` on all data before splitting leaks test statistics into training. Always fit the scaler only on `X_train`.
- **Forgetting `stratify` on multi-class data:** Without it, small classes may vanish from the test set, making evaluation meaningless.
- **Comparing models trained with different preprocessing:** All three models above use the same scaled data — that is fair. Comparing scaled LR to unscaled DT would not be.

{{< collapse summary="Worked solution — preprocessing and training" >}}
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

for name, model in [('LR', LogisticRegression(max_iter=1000)),
                    ('DT', DecisionTreeClassifier(random_state=42)),
                    ('KNN', KNeighborsClassifier(n_neighbors=5))]:
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    print(f'{name}: acc={accuracy_score(y_test, y_pred):.3f}')
```
{{< /collapse >}}

**Extra practice**

1. **Step 1–2 — Exploration:** Load the Wine dataset and display a bar chart of class distribution and the mean value of each feature per class.

{{< pyrepl code="from sklearn.datasets import load_wine\nimport numpy as np\n\nwine = load_wine()\nX, y = wine.data, wine.target\n\n# TODO: print class distribution (count of each class label)\nfor c in np.unique(y):\n    print(f'Class {c}: {(y == c).sum()} samples')\n\n# TODO: print mean of first 3 features per class\nfor c in np.unique(y):\n    means = X[y == c][:, :3].mean(axis=0)\n    print(f'Class {c} feature means (first 3): {means.round(2)}')\n# expected: 3 classes with different feature mean profiles" height="220" >}}

2. **Coding:** Add `cross_val_score` (5-fold) for each of the three models. Report mean ± std. Do the CV scores agree with the single test-set scores?

3. **Challenge:** Add a fourth model: `RandomForestClassifier(n_estimators=100, random_state=42)`. Compare all four models with a bar chart. Does the ensemble beat the individual models?

4. **Variant:** Re-run the pipeline without `StandardScaler`. How much does accuracy change for `LogisticRegression`? For `DecisionTreeClassifier`? Explain why trees are scale-invariant.

5. **Debug:** The code below has a bug — `StandardScaler` is fit on the full dataset before the split, causing data leakage. Find and fix it.

{{< pyrepl code="from sklearn.datasets import load_wine\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\nwine = load_wine()\nX, y = wine.data, wine.target\n\n# BUG: scaler fit on ALL data before splitting (data leakage)\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X_scaled, y, test_size=0.2, random_state=42)\n\nlr = LogisticRegression(max_iter=1000)\nlr.fit(X_train, y_train)\nprint(f'Test accuracy: {lr.score(X_test, y_test):.3f}')\n# TODO: fix by splitting first, then fit scaler only on X_train\n# expected: same or nearly same accuracy, but no leakage" height="200" >}}

6. **Conceptual:** Which model worked best on the Wine dataset in your run? Give one reason why logistic regression might outperform a decision tree on this dataset.

7. **Recall:** In 3 sentences, describe the full ML workflow you executed in this mini-project, from raw data to final model comparison.
