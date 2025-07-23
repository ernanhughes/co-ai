# stephanie/context/context_manager.py
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from stephanie.scoring.score_bundle import ScoreBundle


class ContextManager:
    def __init__(
        self,
        goal: str, 
        max_tokens: int = 8192,
        assembly_fn: Callable[[Dict[str, Any]], str] = None,
        compression_fn: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None,
        scorer_fn: Optional[Callable[[str, Any], ScoreBundle]] = None,
        memory = None,
        logger =None
    ):
        """
        ContextManager wraps a dictionary with introspection and validation
        
        Args:
            goal: Goal text for context prioritization
            max_tokens: Max token limit for assembled context
            assembly_fn: Function to combine components into final prompt
            compression_fn: Function to compress context if needed
            scorer_fn: Function to score context components
            logger: Logger for introspection
        """
        self._data: Dict[str, Any] = {
            "goal": goal,
            "trace": [],
            "metadata": {
                "context_id": str(uuid.uuid4()),
                "start_time": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "token_count": 0,
                "components": {}
            }
        }
        self.max_tokens = max_tokens
        self.assembly_fn = assembly_fn or self.default_assembly
        self.compression_fn = compression_fn or self.default_compression
        self.scorer_fn = scorer_fn
        self.logger = logger

    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self._data[key]

    def __setitem__(self, key, value):
        """Allow dictionary-style assignment"""
        self._data[key] = self._ensure_serializable(value)
        self._update_metadata()
        return self

    def __contains__(self, key):
        """Allow 'in' operator"""
        return key in self._data

    def __call__(self):
        """Make ContextManager itself a context provider"""
        return self._data

    def to_dict(self) -> Dict[str, Any]:
        """Return a pure dictionary for serialization"""
        return self._strip_non_serializable(self._data)

    def update(self, other: Dict[str, Any]):
        """Update with another dictionary"""
        for key, value in other.items():
            self._data[key] = self._ensure_serializable(value)
        self._update_metadata()
        return self

    def add_component(
        self, 
        name: str, 
        content: Any, 
        source: str,
        score: Optional[ScoreBundle] = None,
        priority: float = 1.0
    ):
        """Add a structured component with metadata"""
        # Ensure serializable
        content = self._ensure_serializable(content)
        score = score.to_dict() if score else None
        
        # Add to metadata
        self._data["metadata"]["components"][name] = {
            "name": name,
            "content": content,
            "source": source,
            "score": score,
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update metadata
        self._update_metadata()
        return self

    def assemble(self) -> str:
        """Assemble components into final prompt"""
        try:
            # Score components if scorer is available
            if self.scorer_fn:
                self.score_components()
            
            # Compress if needed
            components = self._data["metadata"]["components"]
            if len(components) > 0:
                components = self.compression_fn(components, self.max_tokens)
            
            # Assemble final prompt
            final_prompt = self.assembly_fn(components)
            self._data["prompt"] = final_prompt
            self._update_metadata()
            return final_prompt
            
        except Exception as e:
            self.logger.log("ContextAssemblyFailed", {
                "error": str(e),
                "components": self._data["metadata"]["components"]
            })
            raise

    def score_components(self):
        """Score components using provided scorer"""
        if not self.scorer_fn:
            return
            
        for name, component in self._data["metadata"]["components"].items():
            try:
                # Score component content
                score = self.scorer_fn(self._data["goal"], component["content"])
                self._data["metadata"]["components"][name]["score"] = score.to_dict()
            except Exception as e:
                self.logger.log("ComponentScoringFailed", {
                    "component": name,
                    "error": str(e)
                })

    def log_action(self, agent, inputs, outputs, description):
        """Log agent actions with introspection"""
        # Ensure serializable inputs/outputs
        self._data["trace"].append({
            "agent": agent.__class__.__name__,
            "inputs": self._strip_non_serializable(inputs),
            "outputs": self._strip_non_serializable(outputs),
            "description": description,
            "timestamp": datetime.utcnow().isoformat()
        })
        self._update_metadata()
        return self

    def _update_metadata(self):
        """Update metadata with latest changes"""
        self._data["metadata"]["last_modified"] = datetime.utcnow().isoformat()
        
        # Count tokens
        token_count = 0
        for key, value in self._data.items():
            if isinstance(value, str):
                token_count += len(value.split()) * 1.5  # Approximate token count
        
        # Update metadata
        self._data["metadata"]["token_count"] = token_count
        return self

    def _ensure_serializable(self, value: Any) -> Any:
        """Ensure value is JSON-serializable"""
        if isinstance(value, torch.Tensor):
            # Convert tensor to list
            return value.tolist()
        if isinstance(value, np.ndarray):
            # Convert numpy array to list
            return value.tolist()
        if isinstance(value, dict):
            # Recursively ensure serializable
            return {k: self._ensure_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            # Process list items
            return [self._ensure_serializable(v) for v in value]
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        return str(value)  # Fallback for other types

    def _strip_non_serializable(self, data: Any) -> Any:
        """Remove non-serializable elements"""
        if isinstance(data, dict):
            return {
                k: self._strip_non_serializable(v) 
                for k, v in data.items() 
                if k not in ["embedding", "logger", "scorer"]
            }
        if isinstance(data, (list, tuple)):
            return [self._strip_non_serializable(v) for v in data]
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data
        return str(data)  # Fallback for non-serializable types

    def default_assembly(self, components: Dict[str, Any]) -> str:
        """Default prompt assembly function"""
        prompt_parts = []
        for name, comp in components.items():
            if isinstance(comp, dict) and "content" in comp:
                prompt_parts.append(f"[{name.upper()}]: {comp['content']}")
            else:
                prompt_parts.append(f"[{name.upper()}]: {comp}")
        return "\n\n".join(prompt_parts)

    def default_compression(self, components: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Default compression strategy"""
        # Implement basic token-based compression
        return {k: v for k, v in components.items() if self._estimate_tokens(v) < max_tokens}

    def _estimate_tokens(self, value: Any) -> int:
        """Estimate token count for a value"""
        if isinstance(value, str):
            return len(value.split()) * 1.5  # Rough estimate
        if isinstance(value, dict):
            return sum(self._estimate_tokens(v) for v in value.values())
        if isinstance(value, list):
            return sum(self._estimate_tokens(v) for v in value)
        return 1  # Minimal for other types

    def _validate_context(self, context: Dict[str, Any]):
        """Ensure context is valid before use"""
        for key, value in context.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                self.logger.log("NaNInContext", {
                    "key": key,
                    "tensor": value.tolist()
                })
                context[key] = [0.0] * len(value)  # Fallback
        return context

    def load_from_dict(self, context: Dict[str, Any]):
        """Load from existing dictionary"""
        self._data = self._validate_context(context)
        self._update_metadata()
        return self