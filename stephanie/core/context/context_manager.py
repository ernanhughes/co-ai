from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from stephanie.scoring.score_bundle import ScoreBundle

@dataclass
class ContextComponent:
    name: str
    content: Any
    source: str
    score: Optional[ScoreBundle] = None
    priority: float = 1.0

@dataclass
class ContextManager:
    max_tokens: int = 8192
    components: Dict[str, ContextComponent] = field(default_factory=dict)
    assembly_fn: Callable[[Dict[str, ContextComponent]], str] = None
    compression_fn: Optional[Callable[[Dict[str, ContextComponent], int], Dict[str, ContextComponent]]] = None
    scoring_fn: Optional[Callable[[str, Any], ScoreBundle]] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def add_component(self, name: str, content: Any, source: str, score: Optional[ScoreBundle] = None, priority: float = 1.0):
        self.components[name] = ContextComponent(name, content, source, score, priority)

    def assemble(self) -> str:
        components = self.components
        if self.compression_fn:
            components = self.compression_fn(components, self.max_tokens)
        prompt = self.assembly_fn(components)
        self.trace.append({"action": "assemble", "tokens": len(prompt.split()), "components": list(components)})
        return prompt

    def score_components(self, query: str):
        if not self.scoring_fn:
            return
        for comp in self.components.values():
            comp.score = self.scoring_fn(query, comp.content)
            self.trace.append({
                "action": "score",
                "component": comp.name,
                "score": comp.score.dict() if hasattr(comp.score, "dict") else str(comp.score)
            })

    def get_context_dict(self) -> Dict[str, Any]:
        return {k: v.content for k, v in self.components.items()}

    def refine(self, feedback: str):
        """Optional: use feedback to update, re-rank, or discard components"""
        # This is a placeholder; you could plug in your MemCube scoring here
        for comp in self.components.values():
            if "error" in feedback and comp.name in feedback:
                comp.priority *= 0.5  # penalize

    def default_assembly_fn(components: Dict[str, ContextComponent]) -> str:
        ordered = sorted(components.values(), key=lambda c: -c.priority)
        return "\n\n".join(f"## {c.name}\n{c.content}" for c in ordered)

    def simple_token_based_compression(components: Dict[str, ContextComponent], max_tokens: int) -> Dict[str, ContextComponent]:
        sorted_comps = sorted(components.items(), key=lambda kv: -kv[1].priority)
        total = 0
        result = {}
        for name, comp in sorted_comps:
            tokens = len(str(comp.content).split())
            if total + tokens <= max_tokens:
                result[name] = comp
                total += tokens
            else:
                break
        return result
