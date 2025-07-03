from abc import ABC, abstractmethod

class Scorable(ABC):
    @property
    @abstractmethod
    def text(self) -> str:
        """Returns the string content to be scored."""
        pass

    @property
    def id(self) -> str:
        """Optional: unique ID if available."""
        return ""
