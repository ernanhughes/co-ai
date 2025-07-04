from abc import ABC, abstractmethod

from stephanie.scoring.scorable import Scorable


class BaseScorer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name or tag for the scorer (e.g. 'svm', 'mrq', 'llm-feedback')"""
        pass

    """ 
    Base interface for any scorer that evaluates a hypothesis given a goal and dimensions.

    Returns:
        A dictionary with dimension names as keys, and for each:
            - score (float)
            - rationale (str)
            - weight (float, optional)
    """
    @abstractmethod
    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> dict:
        raise NotImplementedError("Subclasses must implement the score method.")