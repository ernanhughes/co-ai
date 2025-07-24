# stephanie/context/context_manager.py
import json
import uuid
from datetime import datetime
import numpy as np
import torch
from typing import Any, Dict, Optional
from sqlalchemy import text
from stephanie.models.context_state import ContextStateORM
import sys

class ContextManager:
    def __init__(
        self,
        cfg: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        memory=None,
        logger=None,
        db_table: str = "contexts",
        validate: bool = True,
        save_to_db: bool = True
    ):
        """
        Centralized context manager for pipeline stages
        
        Args:
            goal: Goal text for context prioritization
            run_id: Unique ID for context tracking
            context: Existing context dictionary
            memory: Database connection
            logger: Logging utility
            db_table: Table name for context storage
            validate: Whether to validate context
        """
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.db_table = db_table
        self.validate = validate
        self.save_db = save_to_db
        
        # Initialize context dictionary
        self._data = {
            "trace": [],
            "metadata": {
                "run_id": run_id or str(uuid.uuid4()),
                "start_time": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "token_count": 0,
                "components": {}
            }
        }
        
        # Load from existing context if provided
        if context:
            self._data.update({
                k: v for k, v in context.items() 
                if k not in ["trace", "metadata"]
            })
            self._data["trace"] = context.get("trace", [])
            self._update_metadata()
        
        self.logger.log("ContextManagerInitialized", {
            "run_id": self.run_id,
            "component_count": len(self._data["metadata"]["components"])
        })

    @property
    def run_id(self) -> str:
        """Get context ID from metadata"""
        return self._data["metadata"]["run_id"]

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment with validation"""
        if self._data["metadata"].get("frozen"):
            raise RuntimeError("Context is frozen and cannot be modified.")
        self._data[key] = self._ensure_serializable(value)
        self._update_metadata()
        return self

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator"""
        return key in self._data

    def __call__(self) -> Dict[str, Any]:
        """Make ContextManager itself a context provider"""
        return self._data

    def to_dict(self) -> Dict[str, Any]:
        """Return a pure dictionary for serialization"""
        return self._strip_non_serializable(self._data)


    def freeze(self):
        self._data["metadata"]["frozen"] = True

    def unfreeze(self):
        self._data["metadata"]["frozen"] = False

    def add_component(
        self,
        name: str,
        content: Any,
        source: str,
        score: Optional[Dict] = None,
        priority: float = 1.0
    ) -> None:
        """Add a structured component with metadata"""
        # Ensure content is serializable
        content = self._ensure_serializable(content)
        
        # Score component if scorer is available
        if score is None and hasattr(self, "scorer_fn"):
            try:
                score = self.scorer_fn(self._data["goal"], content)
            except:
                score = {"alignment": 0.5, "clarity": 0.6, "novelty": 0.7}
        
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
            if hasattr(self, "scorer_fn"):
                self.score_components()
            
            # Get components
            components = self._data["metadata"]["components"]
            if not components:
                return self._data["goal"]
            
            # Compress if needed
            components = self._compress_components(components)
            
            # Assemble final prompt
            final_prompt = self._default_assembly(components)
            self._data["prompt"] = final_prompt
            self._update_metadata()
            return final_prompt
            
        except Exception as e:
            self.logger.log("ContextAssemblyFailed", {
                "error": str(e),
                "run_id": self.run_id
            })
            raise

    def score_components(self):
        """Score components using provided scorer"""
        if not hasattr(self, "scorer_fn"):
            return
            
        for name, component in self._data["metadata"]["components"].items():
            try:
                # Score component content
                score = self.scorer_fn(self._data["goal"], component["content"])
                self._data["metadata"]["components"][name]["score"] = score
            except Exception as e:
                self.logger.log("ComponentScoringFailed", {
                    "component": name,
                    "error": str(e)
                })

    def log_action(self, agent, inputs, outputs, description: str):
        """Log agent actions with introspection"""
        # Ensure inputs/outputs are serializable
        self._data["trace"].append({
            "agent": agent.__class__.__name__,
            "inputs": self._strip_non_serializable(inputs),
            "outputs": self._strip_non_serializable(outputs),
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
            "added_keys": list(set(outputs.keys()) - set(inputs.keys()))
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
            if isinstance(value, dict):
                token_count += sum(
                    len(str(v).split()) * 1.5 
                    for v in value.values() 
                    if isinstance(v, str)
                )
        
        self._data["metadata"]["token_count"] = token_count
        return self

    def _ensure_serializable(self, value: Any) -> Any:
        """Ensure value is JSON-serializable"""
        if isinstance(value, torch.Tensor):
            # Convert tensor to list
            if torch.isnan(value).any():
                self.logger.log("TensorContainsNaN", {
                    "tensor": value.tolist(),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype)
                })
                return [0.0] * value.size(0)  # Fallback
            
            return value.tolist()
        
        if isinstance(value, np.ndarray):
            if np.isnan(value).any():
                self.logger.log("ArrayContainsNaN", {
                    "array": value.tolist(),
                    "shape": value.shape
                })
                return [0.0] * value.size
            
            return value.tolist()
        
        if isinstance(value, dict):
            return {k: self._ensure_serializable(v) for k, v in value.items()}
        
        if isinstance(value, (list, tuple)):
            return [self._ensure_serializable(v) for v in value]
        
        # Fallback for other types
        return str(value) if value is not None else value

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

    def _default_assembly(self, components: Dict[str, Any]) -> str:
        """Default prompt assembly function"""
        prompt_parts = []
        for name, comp in components.items():
            if isinstance(comp, dict) and "content" in comp:
                prompt_parts.append(f"[{name.upper()}]: {comp['content']}")
            else:
                prompt_parts.append(f"[{name.upper()}]: {comp}")
        return "\n\n".join(prompt_parts)

    def _compress_components(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Default compression strategy"""
        # Implement basic token-based compression
        return {k: v for k, v in components.items() if self._estimate_tokens(v) < self.max_tokens}

    def _estimate_tokens(self, value: Any) -> int:
        """Estimate token count for a value"""
        if isinstance(value, str):
            return len(value.split()) * 1.5  # Rough estimate
        if isinstance(value, dict):
            return sum(self._estimate_tokens(v) for v in value.values())
        if isinstance(value, list):
            return sum(self._estimate_tokens(v) for v in value)
        return 1  # Minimal for other types

    def save_to_db(self, stage_dict:dict):
        """Save context to database"""
        if not self.save_db:
            return False
            
        # Ensure context is valid
        serializable_context = self._strip_non_serializable(self._data)
        context_size_breakdown = self.context_size_breakdown(serializable_context)
        print(f"Context size breakdown: {context_size_breakdown}")    

        # Save to ORM
        context_orm = ContextStateORM(
            run_id=self.run_id,
            stage_name=stage_dict.get("name", "unknown"),
            context=json.dumps(serializable_context),
            trace=json.dumps(serializable_context["trace"]),
            token_count=serializable_context["metadata"]["token_count"],
            extra_data=json.dumps(stage_dict)
        )
        self.memory.session.add(context_orm)
        self.memory.session.commit()
        return True

    def load_from_db(self, run_id: str):
        """Load context from database"""
        query = text(f"SELECT content FROM {self.db_table} WHERE run_id = :run_id")
        result = self.memory.session.execute(query, {"run_id": run_id}).first()
        
        if result and result.content:
            loaded_context = json.loads(result.content)
            self._data = self._validate_context(loaded_context)
            self._update_metadata()
            return self
        return None

    def _validate_context(self, context: Dict[str, Any]):
        """Ensure context is valid before use"""
        for key, value in context.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                self.logger.log("InvalidContext", {
                    "key": key,
                    "tensor": value.tolist(),
                    "reason": "tensor_contains_nan"
                })
                context[key] = [0.0] * value.size(0)  # Fallback
        
        # Ensure metadata structure
        if "metadata" not in context:
            context["metadata"] = {
                "run_id": str(uuid.uuid4()),
                "token_count": 0,
                "components": {}
            }
        
        return context

    def _compress_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compress context if needed"""
        if not self._exceeds_token_limit(context):
            return context
        
        # Strategy: Drop low-priority components
        components = context["metadata"]["components"]
        prioritized = sorted(
            components.items(),
            key=lambda x: x[1].get("priority", 0.5),
            reverse=True
        )
        
        # Keep top components
        top_components = dict(prioritized[:self.cfg.get("max_components", 5)])
        context["metadata"]["components"] = top_components
        return context

    def _exceeds_token_limit(self, context: Dict[str, Any]) -> bool:
        """Check if context exceeds token limit"""
        token_count = context["metadata"]["token_count"]
        return token_count > self.cfg.get("max_tokens", 8192)

    def load_from_dict(self, context: Dict[str, Any]):
        """Load from existing dictionary"""
        self._data = self._validate_context(context)
        self._update_metadata()
        return self
    
    def extract_memcube(self):
        return {
            "goal": self._data["goal"],
            "components": self._data["metadata"]["components"],
            "summary": self._data.get("summary"),
            "trace": self._data["trace"],
            "created": self._data["metadata"]["start_time"]
        }


    def context_size_breakdown(self, context):
        sizes = {}
        for key, value in context.items():
            try:
                sizes[key] = sys.getsizeof(value)
            except TypeError:
                sizes[key] = "unmeasurable"
        return sorted(sizes.items(), key=lambda x: x[1] if isinstance(x[1], int) else 0, reverse=True)
