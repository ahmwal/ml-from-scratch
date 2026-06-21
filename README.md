# Machine Learning & Statistical Models From Scratch

A pedagogical repository dedicated to implementing foundational machine learning algorithms and statistical models using only core mathematical libraries (e.g., NumPy, SciPy). Designed as a deep-dive reference for understanding the underlying mathematics, optimization routines, and algorithmic mechanics behind modern framework abstractions.

---

## Project Objectives
* **Mathematical Fidelity:** Translating theoretical equations directly into clean, vectorized Python code.
* **No High-Level Frameworks:** Built strictly without `scikit-learn`, `TensorFlow`, or `PyTorch` to ensure zero abstraction hiding.
* **Algorithmic Transparency:** Comprehensive docstrings explaining dimensions, shapes, and optimization choices (e.g., matrix derivatives, gradient descent variants).

---

## Repository Structure & Roadmap

The repository is organized by paradigm. Each directory contains the source implementation, a validation notebook comparing performance against standard benchmarks, and a brief mathematical breakdown.

```text
mlkit
├── 01_regression_models/  # Linear, Logistic, Ridge, Lasso
├── 02_tree_based/         # Decision Trees, Random Forests, Gradient Boosting
├── 03_unsupervised/       # K-Means, PCA, GMMs
├── 04_deep_learning/      # Multilayer Perceptrons, Backpropagation primitives
├── 05_statistics/         # Hypothesis Testing, MLE, Bayesian Inference
└── utils/                 # Metrics, Optimizers, Data splitters from scratch
```


## Sources

This repository is heavily inspired by my ongoing study of a few classic texts:

  The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)

  Pattern Recognition and Machine Learning (Bishop)
