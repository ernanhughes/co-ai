# co_ai/evaluator/base.py
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def judge(self, prompt, output_a, output_b, context:dict):
        pass

    @abstractmethod
    def single_score(self, prompt, output, context:dict):
        pass

