# co_ai/scoring/transforms/regression_tuner.py

from sklearn.linear_model import LinearRegression
import numpy as np


class RegressionTuner:
    """
    Learns to transform MR.Q scores to align with LLM scores dynamically.
    Does not save any state to disk—purely in-memory and real-time.
    """

    def __init__(self, dimension: str, logger=None, min_samples: int = 10):
        self.dimension = dimension
        self.logger = logger
        self.min_samples = min_samples
        self.x = []  # MRQ scores
        self.y = []  # LLM scores
        self.model = None

    def add_example(self, mrq_score: float, llm_score: float):
        """Adds a new example pair and refits the regressor if enough samples."""
        self.x.append(mrq_score)
        self.y.append(llm_score)

        if len(self.x) >= self.min_samples:
            self._fit()

    def _fit(self):
        """Fits a linear regression model to current examples."""
        x_arr = np.array(self.x).reshape(-1, 1)
        y_arr = np.array(self.y)

        self.model = LinearRegression().fit(x_arr, y_arr)

        if self.logger:
            self.logger.log("MRQRegressorFitted", {
                "dimension": self.dimension,
                "count": len(self.x),
                "coef": float(self.model.coef_[0]),
                "intercept": float(self.model.intercept_),
            })

    def transform(self, score: float) -> float:
        """Transforms a score using the fitted regression model if available."""
        if self.model:
            return float(self.model.predict(np.array([[score]]))[0])
        return score
