from typing import Dict, Any, List, Callable, Optional
from abc import ABC

class Protocol(ABC):
    def run(self, input_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method that each protocol must implement.
        """
        pass

ProtocolRecord = Dict[str, Any]