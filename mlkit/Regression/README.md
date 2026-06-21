# Regression Models

This directory houses pure NumPy, "from-scratch" implementations of foundational regression algorithms. The goal is to build these models without high-level wrappers to fully understand the underlying optimization routines, matrix calculus, and evaluation metrics.

---

<details>
<summary><b>1. Linear Regression (OLS, Ridge and Lasso)</b></summary>

> **Status:** `In Progress` | **File:** `linear_regression.py`

A dependency-free implementation of Linear Regression. This class exposes three distinct algorithms to solve for the weight vector $\theta$, supporting both standard Ordinary Least Squares (OLS) and L2 Regularization (Ridge Regression) via the `lmbda` parameter.

### Optimization Methods

| Method | Description | Time Complexity | Best For |
| :--- | :--- | :--- | :--- |
| `fitClosed()` | Exact analytical solution using the Normal Equation. | $O(n^3)$ | Small datasets where matrix inversion is feasible. |
| `fitGradStep()` | Iterative optimization using Batch Gradient Descent. | $O(k \cdot n^2)$ | Large datasets where computing $(X^T X)^{-1}$ is expensive. |
| `fitNaiveStep()` | A greedy coordinate-descent search. | Variable | Pedagogical proofs and hypothesis testing. |

#### The "Naive Step" Hypothesis
The `fitNaiveStep` method was built to test a specific hypothesis: *Under the assumption of non-collinearity, independently adjusting each parameter to greedily minimize the Sum of Squared Residuals (SSR) will eventually converge to the global optimum.* The implementation confirms this. However, because it acts as a greedy search—moving strictly in the immediate direction that minimizes SSR for one parameter at a time—it is computationally inefficient and significantly slower than both the gradient descent and closed-form solutions.

As the SSR can be graphed as a continuous function, this search can then be skipped by using the first order (the Jacobian) minima equation and second order (the Hessian) derivative concavity test to reach the OLS closed-form solution (under the assumption of a non-collinear parameter set), or by using guided search through gradient descent through getting the derivative of the cost function with respect to the parameters of the linear regression model.

### 📊 Performance Comparison

*(Placeholder: Add your performance benchmark here comparing the execution time and final SSR of `fitClosed`, `fitGradStep`, and `fitNaiveStep`.)*

`[Insert Benchmark Plot Here: e.g., ![Convergence Comparison](./assets/convergence_plot.png)]`

### The Mathematics Under the Hood

**1. Closed-Form Solution (Normal Equation)**
Calculates the exact coefficients by minimizing the cost function algebraically. If `lmbda > 0`, it applies L2 regularization (Ridge), where $I$ is the identity matrix. [Derivation](../notes)

$$\beta = (X^T X + \lambda I)^{-1} X^T y$$

Where 
$$\beta$$ is the coefficient vector, 
$$X$$ is the independant parameter matrix, 
$$\lambda$$ is regularization penalty scaler and
$$y$$ is the dependant parameter vector

**2. Gradient Descent**
Iteratively updates the weights by taking steps in the direction of the negative gradient of the Mean Squared Error loss function. [Derivation](../notes)

$$\nabla_\beta J(\beta) = -\alpha(\frac{2}{n} X^T (y - X\beta) + 2\lambda\beta)$$

Where
$$\nabla_\beta J(\beta)$$ is the gradient of our coefficients at this step, 
$$\beta$$ is the coefficient vector,
$$X$$ is the independant parameter matrix, 
$$\lambda$$ is regularization penalty scaler, 
$$y$$ is the dependant parameter vector, 
$$n$$ is the number of samples, 
$$\alpha$$ is the learning rate

**3. Model Evaluation ($R^2$ and F-Statistic)**
Evaluates goodness-of-fit by comparing the Sum of Squared Residuals of the fitted model ($SSR_{fit}$) against a baseline model ($SSR_{mean}$).
This just compares the variance explained by the model to the total variance.

$$R^2 = \frac{SSR_{mean} - SSR_{fit}}{SSR_{mean}}$$

### Usage Example

```python
import numpy as np
from linear_regression import LinearRegression

# 1. Generate some synthetic linear data
np.random.seed(42)
X = np.random.rand(100, 2)
y = 3.5 * X[:, 0] + 1.2 * X[:, 1] + 5.0 + np.random.randn(100) * 0.1

# 2. Instantiate and Fit
model = LinearRegression(num_params=2, seed=42)

# Option A: Closed form with L2 regularization
model.fitClosed(X, y, lmbda=0.1)

# Option B: Gradient Descent
# model.fitGradStep(X, y, step=0.01, convergence_thresh=1e-4, lmbda=0.1)

# Option C: Naive Greedy Step
# model.fitNaiveStep(X, y, step=0.001)

# 3. Evaluate
stats = model.getRSquared()
print(f"R-squared: {stats['r2']}")
print(f"F-statistic: {stats['F']}")
```

</details>

> **Status:** `Planned`

*Implementation details, mathematical derivations, and more code coming soon.*
