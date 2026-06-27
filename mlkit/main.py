import pandas as pd
from Regression.LinearRegression import LinearRegression
from utils.scaling import zscoreStandardize
from utils.stats import meanSquaredError

seed = 742


def main():
    data = pd.read_csv("./data/combined_diabetes.csv")
    train = data.sample(frac=0.8, random_state=seed)
    test = data.drop(train.index).iloc[:, :-1]
    test_y = data.drop(train.index).iloc[:, -1:]
    test_y = (
        test_y.to_numpy().flatten()
        if test_y.to_numpy().shape[-1] == 1
        else test_y.to_numpy()
    )
    x = train.iloc[:, :-1].to_numpy()
    x_scaled = zscoreStandardize(x)
    y = train.iloc[:, -1:].to_numpy()

    x_test = test.to_numpy()
    x_test_scaled = zscoreStandardize(x_test)

    print(train.to_string(), test.to_string(), test_y)

    myregClosedSol = LinearRegression(test.to_numpy().shape[-1], seed)
    myregClosedSol.fitClosed(x_scaled, y, lmbda=1)
    closedpred = myregClosedSol.predict(x_test_scaled)
    closedcoef = myregClosedSol.coefficients

    # myregNaiveDescent = LinearRegression(test.to_numpy().shape[-1], seed)
    # myregNaiveDescent.fitNaiveStep(x, y)
    # naivepred = myregNaiveDescent.predict(x_test)
    # naivecoef = myregNaiveDescent.coefficients

    myregGradDescent = LinearRegression(test.to_numpy().shape[-1], seed)
    myregGradDescent.fitGradStep(x_scaled, y, lmbda=0)
    gradpred = myregGradDescent.predict(x_test_scaled)
    gradcoef = myregGradDescent.coefficients

    print(
        closedpred,
        closedcoef,
        # naivepred,
        # naivecoef,
        gradpred,
        gradcoef,
        test_y,
        myregClosedSol.getRSquared(),
        "\n",
        "MSE CLOSED PRED\n",
        meanSquaredError(test_y, closedpred),
        "\n",
        "MSE GRAD PRED\n",
        meanSquaredError(test_y, gradpred),
    )


if __name__ == "__main__":
    main()
