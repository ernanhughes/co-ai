# co_ai/scoring/calculations/base_calculator.py
from abc import ABC, abstractmethod


class BaseScoreCalculator(ABC):
    @abstractmethod
    def calculate(self, results: dict) -> float:
        """
        Given a dict of dimension results (each with score, weight), return a single float score.
        """
        pass
