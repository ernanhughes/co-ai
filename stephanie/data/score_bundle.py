import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from stephanie.data.score_result import ScoreResult


@dataclass
class ScoreBundle:
    results: dict[str, ScoreResult] = field(default_factory=dict)
    
    def __init__(self, results: dict[str, ScoreResult], dimension_config: Dict = None):
        from stephanie.scoring.calculations.mars_calculator import MARSCalculator
        self.results = results
        self.dimension_config = dimension_config
        self.mars_calculator = MARSCalculator(dimension_config)
        self.mars_analysis = None
    
    def analyze_agreement(self) -> Dict:
        """Perform MARS analysis on the score bundle"""
        self.mars_analysis = self.mars_calculator.calculate(self)
        return self.mars_analysis
    
    def aggregate(self, use_mars: bool = True) -> Union[float, Dict]:
        """
        Aggregate scores, with option to use MARS analysis for enhanced aggregation
        
        If use_mars is True, returns a dictionary with:
        - score: the final aggregated score
        - agreement: agreement metrics
        - uncertainty: uncertainty score
        
        If use_mars is False, returns just the float score (legacy behavior)
        """
        if use_mars and self.results:
            return self.analyze_agreement()
        else:
            # Fall back to simple weighted average for legacy compatibility
            total = sum(r.score * getattr(r, "weight", 1.0) for r in self.results.values())
            weight_sum = sum(getattr(r, "weight", 1.0) for r in self.results.values())
            return round(total / weight_sum, 2) if weight_sum else 0.0
    
    def get(self, dimension: str) -> Optional[ScoreResult]:
        return self.results.get(dimension)
    
    def to_dict(self, include_mars: bool = True) -> Dict[str, Any]:
        base_dict = {dim: result.to_dict() for dim, result in self.results.items()}
        
        if include_mars and self.mars_analysis is None and self.results:
            self.analyze_agreement()
            
        if include_mars and self.mars_analysis:
            base_dict["mars_analysis"] = self.mars_analysis
            
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], dimension_config: Dict = None) -> "ScoreBundle":
        """Reconstruct a ScoreBundle from a dictionary"""
        # Extract mars_analysis if present
        mars_analysis = data.pop("mars_analysis", None)
        
        results = {
            dim: ScoreResult.from_dict(score_data)
            for dim, score_data in data.items()
            if isinstance(score_data, dict)
        }
        
        bundle = cls(results=results, dimension_config=dimension_config)
        
        # If mars_analysis was included, store it
        if mars_analysis:
            bundle.mars_analysis = mars_analysis
            
        return bundle
    
    def merge(self, other: "ScoreBundle") -> "ScoreBundle":
        """Merge two bundles, preferring `self` values but including all from both"""
        merged = dict(self.results)
        for dim, result in other.results.items():
            if dim not in merged:
                merged[dim] = result
        return ScoreBundle(merged, dimension_config=self.dimension_config)
    
    def to_json(self, stage: str, include_mars: bool = True):
        aggregate = self.aggregate(use_mars=include_mars)
        return {
            "stage": stage,
            "dimensions": self.to_dict(include_mars=include_mars),
            "final_score": aggregate["score"] if isinstance(aggregate, dict) else aggregate,
            **({"mars_analysis": aggregate} if isinstance(aggregate, dict) else {})
        }
    
    def to_orm(self, evaluation_id: int):
        from stephanie.models.score import ScoreORM
        # First, create regular score ORM objects
        regular_scores = [
            ScoreORM(
                evaluation_id=evaluation_id,
                dimension=r.dimension,
                score=r.score,
                weight=r.weight,
                rationale=r.rationale,
                source=r.source,
                target_type=r.target_type,
                prompt_hash=r.prompt_hash,
                energy=r.energy,
                q_value=r.q_value,
                state_value=r.state_value,
                policy_logits=r.policy_logits,
                uncertainty=r.uncertainty,
                entropy=r.entropy,
                advantage=r.advantage,
            )
            for r in self.results.values()
        ]
        
        # If we have MARS analysis, add it as a special score
        if self.mars_analysis:
            mars_result = ScoreResult(
                dimension=list(self.results.values())[0].dimension,
                score=self.mars_analysis["score"],
                rationale=self.mars_analysis["explanation"],
                source="mars",
                uncertainty=self.mars_analysis["uncertainty"],
                weight=1.0
            )
            
            mars_score = ScoreORM(
                evaluation_id=evaluation_id,
                dimension=mars_result.dimension,
                score=mars_result.score,
                weight=mars_result.weight,
                rationale=mars_result.rationale,
                source=mars_result.source,
                target_type="meta",
                prompt_hash="",
                uncertainty=mars_result.uncertainty
            )
            regular_scores.append(mars_score)
            
        return regular_scores
    
    def __repr__(self):
        if self.mars_analysis:
            return f"<ScoreBundle(score={self.mars_analysis['score']:.2f}, " \
                   f"agreement={self.mars_analysis['agreement']['agreement_score']:.2f}, " \
                   f"uncertainty={self.mars_analysis['uncertainty']:.2f})>"
        else:
            summary = ", ".join(
                f"{dim}: {res.score}" for dim, res in self.results.items()
            )
            return f"<ScoreBundle({summary})>"
    
    def to_report(self, title: str = "Score Report") -> str:
        lines = [f"## {title}", ""]
        
        # Add MARS analysis if available
        if self.mars_analysis is None and self.results:
            self.analyze_agreement()
            
        if self.mars_analysis:
            lines.append(f"### Meta-Analysis (MARS)")
            lines.append(f"- **Aggregate Score**: `{self.mars_analysis['score']:.2f}`")
            lines.append(f"- **Agreement Score**: `{self.mars_analysis['agreement']['agreement_score']:.3f}`")
            lines.append(f"- **Uncertainty**: `{self.mars_analysis['uncertainty']:.3f}`")
            lines.append(f"- **Trust Reference**: `{self.mars_analysis['trust_reference']}`")
            lines.append(f"- **Explanation**: {self.mars_analysis['explanation']}")
            lines.append("")
        
        # Add individual dimension scores
        for dim, result in self.results.items():
            lines.append(f"### Dimension: `{dim}`")
            lines.append(f"- **Score**: `{result.score:.4f}`")
            lines.append(f"- **Weight**: `{result.weight:.2f}`")
            lines.append(f"- **Source**: `{result.source}`")
            lines.append(f"- **Target Type**: `{result.target_type}`")
            lines.append(f"- **Prompt Hash**: `{result.prompt_hash}`")
            if result.rationale:
                lines.append(f"- **Rationale**: {result.rationale}")
            # SICQL-specific fields
            if result.energy is not None:
                lines.append(f"- **Energy**: `{result.energy:.4f}`")
            if result.q_value is not None:
                lines.append(f"- **Q-Value**: `{result.q_value:.4f}`")
            if result.state_value is not None:
                lines.append(f"- **State Value**: `{result.state_value:.4f}`")
            if result.policy_logits is not None:
                logits_str = ", ".join(
                    f"{x:.4f}" for x in result.policy_logits
                )
                lines.append(f"- **Policy Logits**: [{logits_str}]")
            if result.uncertainty is not None:
                lines.append(f"- **Uncertainty**: `{result.uncertainty:.4f}`")
            if result.entropy is not None:
                lines.append(f"- **Entropy**: `{result.entropy:.4f}`")
            if result.advantage is not None:
                lines.append(f"- **Advantage**: `{result.advantage:.4f}`")
            lines.append("")  # Empty line between dimensions
        
        return "\n".join(lines)