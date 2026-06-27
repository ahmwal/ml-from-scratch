import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
from Regression.LinearRegression import (
    gradientDescentLinearRegression as baseGradRegression,
)
from Regression.LinearRegression import (
    lassoGradientDescentSubGradientLinearRegression as baseGradSubgradientLassoRegression,
)
from Regression.LinearRegression import (
    naiveDescentLinearRegression as naiveRegression,
)
from Regression.LinearRegression import (
    normalEquationLinearRegression as normalRegression,
)
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import LinearRegression as SkLinear
from sklearn.linear_model import Ridge as SkRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from utils.stats import getFStatistic, meanSquaredError

seed = 742


# ==========================================
# 2. EVALUATION FRAMEWORK
# ==========================================
def benchmark_models(models_dict, X_train, y_train, X_test, y_test):
    """
    Trains, predicts, and evaluates a dictionary of models.
    """
    results = {}

    for name, model in models_dict.items():
        print(f"Evaluating {name}...")

        # Start tracking memory and time
        tracemalloc.start()
        start_time = time.perf_counter()

        # Train the model
        model.fit(X_train, y_train)

        # Stop tracking
        end_time = time.perf_counter()
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate Metrics
        try:
            mse = meanSquaredError(y_test, predictions)
            f = getFStatistic(y_test, predictions, X_train.shape[0], model.coefficients)
            print(model.coefficients)
        except Exception as e:
            mse = mean_squared_error(y_test, predictions)
            f = getFStatistic(y_test, predictions, X_train.shape[0], model.coef_)
            print(model.coef_)

        r2 = r2_score(y_test, predictions)

        time_taken = end_time - start_time
        mem_used_kb = peak_memory / 1024  # Convert to Kilobytes

        results[name] = {
            "MSE": mse,
            "R2": r2,
            "F": f,
            "Time (s)": time_taken,
            "Memory (KB)": mem_used_kb,
            "Predictions": predictions,
            "Model_Instance": model,  # Kept in case we want to plot internal loss history
        }

    return results


# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================
def plot_bar_comparisons(results):
    """Plots bar charts for MSE, R2, Time, and Memory usage."""
    names = list(results.keys())
    mse_vals = [results[n]["MSE"] for n in names]
    r2_vals = [results[n]["R2"] for n in names]
    time_vals = [results[n]["Time (s)"] for n in names]
    mem_vals = [results[n]["Memory (KB)"] for n in names]
    f_vals = [results[n]["F"] for n in names]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    fig.suptitle("Regression Algorithms Comparison", fontsize=16)

    # 1. Mean Squared Error (Lower is better)
    axes[0, 0].bar(names, mse_vals, color="skyblue", edgecolor="black")
    axes[0, 0].set_title("Mean Squared Error (Lower is better)")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. R-Squared (Higher is better, max 1.0)
    axes[0, 1].bar(names, r2_vals, color="lightgreen", edgecolor="black")
    axes[0, 1].set_title("Goodness of Fit: R² Score (Closer to 1 is better)")
    axes[0, 1].set_ylabel("R²")
    axes[0, 1].set_ylim(min(0, min(r2_vals) * 1.1), 1.05)  # handle negative R2
    axes[0, 1].tick_params(axis="x", rotation=45)

    # 3. Convergence Time
    axes[1, 0].bar(names, time_vals, color="salmon", edgecolor="black")
    axes[1, 0].set_title("Training Time (Lower is better)")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 4. Memory Used
    axes[1, 1].bar(names, mem_vals, color="orchid", edgecolor="black")
    axes[1, 1].set_title("Peak Memory Usage (Lower is better)")
    axes[1, 1].set_ylabel("Kilobytes")
    axes[1, 1].tick_params(axis="x", rotation=45)

    # 5. F-Statistic
    axes[0, 2].bar(names, mem_vals, color="orchid", edgecolor="black")
    axes[0, 2].set_title("F-Statistic (Higher is better)")
    axes[0, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_goodness_of_fit(results, y_test, num_samples=50):
    """Plots True Values vs Predictions for a subset of the data."""
    plt.figure(figsize=(12, 6))

    # Plot true values
    plt.plot(
        y_test[:num_samples],
        label="True Values",
        color="black",
        marker="o",
        linestyle="dashed",
        linewidth=2,
        markersize=8,
    )

    # Plot predictions for each model
    for name, data in results.items():
        plt.plot(
            data["Predictions"][:num_samples],
            label=f"{name} Preds",
            marker="x",
            alpha=0.7,
        )

    plt.title("Predictions vs True Values (First 50 Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_convergence_curves(results):
    """
    Plots the loss over epochs IF your models save it.
    Assumes your models have a property like `self.loss_history`.
    """
    plt.figure(figsize=(10, 6))
    plotted_any = False

    for name, data in results.items():
        model = data["Model_Instance"]
        # Check if your custom model saves cost/loss history during gradient descent
        if hasattr(model, "loss_history") or hasattr(model, "cost_history"):
            history = getattr(model, "loss_history", getattr(model, "cost_history", []))
            if len(history) > 0:
                plt.plot(history, label=name)
                plotted_any = True

    if plotted_any:
        plt.title("Convergence Time: Loss Over Iterations")
        plt.xlabel("Iterations / Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print(
            "Note: Skipping convergence plot. Models do not have 'loss_history' or 'cost_history' attributes."
        )


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Generate some synthetic regression data for testing
    print("Generating synthetic data...")
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.5, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    models_to_test = {
        "Scikit Base Linear": SkLinear(),
        "Scikit SGD Ridge (L2)": SkRidge(alpha=1.0),
        "Scikit SGD Lasso (L1)": SkLasso(alpha=1.0),
        "mlkit NE Linear": normalRegression(seed),
        "mlkit basic GD Linear": baseGradRegression(seed),
        "mlkit NE Ridge Linear": normalRegression(seed, 1.0),
        "mlkit basic GD Ridge Linear": baseGradRegression(seed, 1.0),
        "mlkit basic GD Lasso Linear": baseGradSubgradientLassoRegression(seed, 1.0),
        "mlkit naive Linear": naiveRegression(seed),
        "mlkit naive Ridge Linear": naiveRegression(seed, 1),
    }

    # Run benchmarks
    results = benchmark_models(models_to_test, X_train, y_train, X_test, y_test)

    # Generate Graphs
    print("\nGenerating visual reports...")
    plot_bar_comparisons(results)
    plot_goodness_of_fit(results, y_test)
    plot_convergence_curves(results)
