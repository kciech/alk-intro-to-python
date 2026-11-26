"""Group 6 (w03) - model_judge: checks a consultant's claim that a polynomial trend forecasts our daily revenue well.

The model judged here: revenue as a polynomial curve of time,
np.polyfit(years, revenue, degree) with years = day_number / 365.
We compare the error on the days the curve was fitted on (train) with the error on the
days AFTER them (test), and random cross-validation with time cross-validation.

Authors: Alex, Lukasz, Adam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, TimeSeriesSplit

from ml_base import Analysis


class ModelJudge(Analysis):
    """Judges polynomial trend models of daily revenue: training error vs error on the next months.

    Usage:
        judge = ModelJudge(daily).fit()
        judge.results_            # train_mae and test_mae for degree 1 ... max_degree
        judge.best_degree()       # the degree with the lowest test_mae
        judge.cv_mae(3, method="time")
        judge.plot(3)

    Args:
        daily: DataFrame with a date index and the columns revenue and day_number (load_daily())
        last_train_day: the last day used for fitting; the days after it are the test days
        max_degree: fit every degree 1 ... max_degree (an integer >= 1)

    Raises:
        TypeError: if daily is not a DataFrame
        ValueError: if a column is missing, max_degree < 1, last_train_day is not a date,
            or there are no training days or no test days
    """
    required_columns = ["revenue", "day_number"]

    def __init__(self, daily, last_train_day="2011-09-30", max_degree=8):
        super().__init__(daily)
        if not isinstance(max_degree, (int, np.integer)) or isinstance(max_degree, bool):
            raise ValueError(f"max_degree must be a whole number, got {max_degree!r}")
        if max_degree < 1:
            raise ValueError(f"max_degree must be at least 1, got {max_degree}")
        try:
            self.last_train_day = pd.Timestamp(last_train_day)
        except (ValueError, TypeError):
            raise ValueError(f"last_train_day must be a date like '2011-09-30', got {last_train_day!r}")
        self.max_degree = int(max_degree)
        self.data = self.data.sort_index()
        self.years = self.data["day_number"].to_numpy() / 365
        self.revenue = self.data["revenue"].to_numpy()
        self.is_train = (self.data.index <= self.last_train_day)
        if self.is_train.sum() == 0 or (~self.is_train).sum() == 0:
            raise ValueError("last_train_day must leave at least one training day and one test day")
        self.coefs_ = {}

    def fit(self):
        """Fit the curve for every degree 1 ... max_degree on the days up to last_train_day.

        For each degree, computes the mean absolute error (MAE, GBP) on the training days
        (train_mae) and on the days after last_train_day (test_mae).
        Stores a DataFrame (index degree, columns train_mae, test_mae, whole GBP) in self.results_
        and the polynomial coefficients in self.coefs_ (dict degree -> array).

        Returns:
            self
        """
        x_train, y_train = self.years[self.is_train], self.revenue[self.is_train]
        x_test, y_test = self.years[~self.is_train], self.revenue[~self.is_train]
        rows = []
        for degree in range(1, self.max_degree + 1):
            coefs = np.polyfit(x_train, y_train, degree)
            self.coefs_[degree] = coefs
            rows.append({
                "degree": degree,
                "train_mae": round(np.mean(np.abs(y_train - np.polyval(coefs, x_train)))),
                "test_mae": round(np.mean(np.abs(y_test - np.polyval(coefs, x_test)))),
            })
        self.results_ = pd.DataFrame(rows).set_index("degree")
        return self

    def best_degree(self):
        """The degree with the lowest test_mae (an int).

        Raises:
            RuntimeError: if fit() was not called
        """
        self.check_fitted()
        return int(self.results_["test_mae"].idxmin())

    def cv_mae(self, degree, method="time"):
        """Mean MAE (whole GBP) of 5-fold cross-validation of the curve of this degree on ALL days.

        Args:
            degree: polynomial degree, a whole number from 1 to max_degree
            method: "time" -> TimeSeriesSplit(5): always train on the past, test on the next days;
                    "random" -> KFold(5, shuffle=True, random_state=0): test days are mixed between training days

        Raises:
            RuntimeError: if fit() was not called
            ValueError: if method is not "time" or "random", or the degree is not valid
        """
        self.check_fitted()
        self._check_degree(degree)
        if method == "time":
            splitter = TimeSeriesSplit(n_splits=5)
        elif method == "random":
            splitter = KFold(n_splits=5, shuffle=True, random_state=0)
        else:
            raise ValueError(f"method must be 'time' or 'random', got {method!r}")
        errors = []
        for train_idx, test_idx in splitter.split(self.years):
            coefs = np.polyfit(self.years[train_idx], self.revenue[train_idx], degree)
            predicted = np.polyval(coefs, self.years[test_idx])
            errors.append(np.mean(np.abs(self.revenue[test_idx] - predicted)))
        return round(np.mean(errors))

    def predict(self, degree):
        """Revenue predicted by the fitted curve of this degree for every day of the data (NumPy array).

        Raises:
            RuntimeError: if fit() was not called
            ValueError: if the degree is not valid
        """
        self.check_fitted()
        self._check_degree(degree)
        return np.polyval(self.coefs_[degree], self.years)

    def plot(self, degree, ax=None):
        """Daily revenue (dots) + the fitted curve of this degree + a dashed line at last_train_day.

        Returns:
            the matplotlib Axes

        Raises:
            RuntimeError: if fit() was not called
            ValueError: if the degree is not valid
        """
        self.check_fitted()
        self._check_degree(degree)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(self.data.index, self.revenue, s=8, color="grey", alpha=0.6, label="daily revenue")
        ax.plot(self.data.index, self.predict(degree), color="tab:red", linewidth=2,
                label=f"degree {degree} curve")
        ax.axvline(self.last_train_day, color="black", linestyle="--", label="last training day")
        train_mae, test_mae = self.results_.loc[degree]
        ax.set(title=f"Degree {degree}: train MAE £{train_mae:,} · test MAE £{test_mae:,}",
               xlabel="date", ylabel="revenue (GBP)")
        low, high = np.percentile(self.revenue, [0, 100])
        ax.set_ylim(min(0, low), high * 1.3)          # a wild curve should not squash the data
        ax.legend(loc="upper left")
        return ax

    def _check_degree(self, degree):
        """Raise ValueError unless degree is a whole number from 1 to max_degree."""
        if (not isinstance(degree, (int, np.integer)) or isinstance(degree, bool)
                or not 1 <= degree <= self.max_degree):
            raise ValueError(f"degree must be a whole number from 1 to {self.max_degree}, got {degree!r}")
