# co_ai/scoring/calculations/weighted_average.py
from co_ai.scoring.calculations.base_calculator import BaseScoreCalculator

class WeightedAverageCalculator(BaseScoreCalculator):
    def calculate(self, results: dict) -> float:
        total = sum(r["score"] * r.get("weight", 1.0) for r in results.values())
        weight_sum = sum(r.get("weight", 1.0) for r in results.values())
        return round(total / weight_sum, 2) if weight_sum else 0.0
