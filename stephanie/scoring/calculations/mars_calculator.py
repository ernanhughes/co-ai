import numpy as np
from scipy import stats
from statistics import mean, stdev
from typing import Dict, List, Tuple, Optional

from stephanie.scoring.calculations.base_calculator import BaseScoreCalculator
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult


class MARSCalculator(BaseScoreCalculator):
    """Advanced meta-scoring that analyzes agreement across different scoring models"""
    
    def __init__(self, dimension_config: Dict = None):
        """
        dimension_config example:
        {
            "helpfulness": {
                "scale": "0-100",
                "trust_references": ["llm", "human"],
                "expected_variance": 0.15
            },
            "truthfulness": {
                "scale": "0-1",
                "trust_references": ["fact_checker", "human"],
                "expected_variance": 0.1
            }
        }
        """
        self.dimension_config = dimension_config or {}
        self.historical_agreement = {}  # Track agreement patterns over time
    
    def _normalize_score(self, score: ScoreResult, dimension: str) -> float:
        """Normalize scores to 0-1 scale based on dimension configuration"""
        config = self.dimension_config.get(dimension, {})
        scale = config.get("scale", "0-1")
        
        if scale == "0-1":
            return score.score
        elif scale == "0-100":
            return score.score / 100.0
        elif scale == "categorical":
            # Map categorical scores to numeric (would need more sophisticated handling)
            category_map = {"poor": 0.2, "fair": 0.4, "good": 0.7, "excellent": 0.9}
            return category_map.get(score.score, 0.5)
        else:
            # Default normalization based on historical data
            return self._historical_normalization(score, dimension)
    
    def _historical_normalization(self, score: ScoreResult, dimension: str) -> float:
        """Normalize using historical data if config doesn't specify scale"""
        # In a real implementation, this would use actual historical data
        # This is a placeholder for demonstration
        return score.score / 100.0 if score.score > 10 else score.score
    
    def _calculate_agreement_metrics(self, normalized_scores: List[float]) -> Dict:
        """Calculate advanced agreement metrics beyond simple std deviation"""
        n = len(normalized_scores)
        if n < 2:
            return {"agreement_score": 1.0, "std_dev": 0.0}
        
        # Calculate standard deviation (on normalized 0-1 scale)
        std_dev = stdev(normalized_scores)
        
        # Fleiss' Kappa for inter-rater agreement (simplified version)
        # Measures agreement beyond chance
        observed_agreement = 1.0 - std_dev
        expected_agreement = 1.0 / n  # Simplified assumption
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement) if expected_agreement < 1 else 1.0
        
        # Entropy-based agreement measure (lower entropy = higher agreement)
        histogram, _ = np.histogram(normalized_scores, bins=5, range=(0, 1), density=True)
        entropy = stats.entropy(histogram + 1e-10)  # Add small value to avoid log(0)
        normalized_entropy = entropy / np.log(len(histogram))
        entropy_agreement = 1.0 - normalized_entropy
        
        return {
            "std_dev": std_dev,
            "fleiss_kappa": max(0.0, min(1.0, kappa)),  # Clamp to [0,1]
            "entropy_agreement": entropy_agreement,
            "agreement_score": (observed_agreement + entropy_agreement) / 2
        }
    
    def _identify_conflict_clusters(self, scores: List[ScoreResult], 
                                  normalized_scores: List[float]) -> Dict:
        """Identify clusters of agreement rather than just max vs min"""
        # Simple k-means for 2 clusters (could be enhanced)
        if len(normalized_scores) < 3:
            return {
                "clusters": [{"members": [scores[0].source], "mean": normalized_scores[0]}],
                "primary_conflict": None,
                "cluster_count": 1
            }
        
        # Very simple clustering (for demonstration - would use actual k-means in production)
        midpoint = mean(normalized_scores)
        cluster1 = [s for s, ns in zip(scores, normalized_scores) if ns <= midpoint]
        cluster2 = [s for s, ns in zip(scores, normalized_scores) if ns > midpoint]
        
        clusters = []
        if cluster1:
            cluster_mean = mean([ns for ns in normalized_scores if ns <= midpoint])
            clusters.append({
                "members": [s.source for s in cluster1],
                "mean": cluster_mean,
                "size": len(cluster1)
            })
        if cluster2:
            cluster_mean = mean([ns for ns in normalized_scores if ns > midpoint])
            clusters.append({
                "members": [s for s in cluster2],
                "mean": cluster_mean,
                "size": len(cluster2)
            })
        
        # Identify primary conflict as the largest gap between clusters
        primary_conflict = None
        max_gap = 0
        for i in range(len(clusters)-1):
            gap = abs(clusters[i]["mean"] - clusters[i+1]["mean"])
            if gap > max_gap:
                max_gap = gap
                primary_conflict = (clusters[i]["members"], clusters[i+1]["members"])
        
        return {
            "clusters": clusters,
            "primary_conflict": primary_conflict,
            "cluster_count": len(clusters),
            "max_gap": max_gap
        }
    
    def _determine_trust_reference(self, dimension: str, 
                                 scores: List[ScoreResult]) -> Tuple[str, float]:
        """Determine the most reliable scorer based on dimension config and historical data"""
        config = self.dimension_config.get(dimension, {})
        trust_refs = config.get("trust_references", ["human", "llm"])
        
        # First try configured trust references
        for ref in trust_refs:
            for score in scores:
                if score.source == ref:
                    return ref, score.score
        
        # If no configured reference available, use the median scorer
        median_idx = len(scores) // 2
        return scores[median_idx].source, scores[median_idx].score
    
    def _calculate_uncertainty(self, agreement_metrics: Dict, 
                             cluster_info: Dict, dimension: str) -> float:
        """Calculate uncertainty score based on agreement metrics"""
        config = self.dimension_config.get(dimension, {})
        expected_variance = config.get("expected_variance", 0.15)
        
        # Higher uncertainty when agreement is low OR when there are multiple clusters
        uncertainty = 1.0 - agreement_metrics["agreement_score"]
        if cluster_info["cluster_count"] > 1:
            uncertainty = max(uncertainty, cluster_info["max_gap"])
        
        # Adjust based on expected variance for this dimension
        return min(1.0, uncertainty / max(0.01, expected_variance))
    
    def calculate(self, bundle: ScoreBundle) -> Dict:
        """Calculate MARS metrics and return enhanced aggregate information"""
        results = list(bundle.results.values())
        if not results:
            return {
                "score": 0.0,
                "agreement": {"agreement_score": 0.0},
                "uncertainty": 1.0
            }
        
        dimension = results[0].dimension  # All should have same dimension
        normalized_scores = []
        sources = []
        
        # Normalize all scores to 0-1 scale
        for result in results:
            normalized = self._normalize_score(result, dimension)
            normalized_scores.append(normalized)
            sources.append(result.source)
        
        # Calculate agreement metrics
        agreement_metrics = self._calculate_agreement_metrics(normalized_scores)
        
        # Identify conflict clusters
        cluster_info = self._identify_conflict_clusters(results, normalized_scores)
        
        # Determine trust reference
        trust_ref, _ = self._determine_trust_reference(dimension, results)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(agreement_metrics, cluster_info, dimension)
        
        # Calculate weighted score (giving more weight to reliable scorers)
        weights = self._calculate_scorer_weights(results, trust_ref, agreement_metrics)
        weighted_score = sum(ns * w for ns, w in zip(normalized_scores, weights)) / sum(weights) if weights else 0.0
        
        # Store in historical data for trend analysis
        self._update_historical_data(dimension, agreement_metrics, cluster_info)
        
        return {
            "score": round(weighted_score * 100, 2),  # Return on 0-100 scale for consistency
            "agreement": {
                "agreement_score": round(agreement_metrics["agreement_score"], 3),
                "std_dev": round(agreement_metrics["std_dev"], 3),
                "fleiss_kappa": round(agreement_metrics["fleiss_kappa"], 3),
                "entropy_agreement": round(agreement_metrics["entropy_agreement"], 3),
                "cluster_count": cluster_info["cluster_count"],
                "primary_conflict": cluster_info["primary_conflict"]
            },
            "trust_reference": trust_ref,
            "uncertainty": round(uncertainty, 3),
            "scorer_weights": dict(zip(sources, [round(w, 3) for w in weights])),
            "explanation": self._generate_explanation(
                agreement_metrics, cluster_info, trust_ref, uncertainty
            )
        }
    
    def _calculate_scorer_weights(self, results: List[ScoreResult], 
                                trust_ref: str, agreement_metrics: Dict) -> List[float]:
        """Calculate dynamic weights for each scorer based on reliability"""
        weights = []
        for result in results:
            # Base weight on agreement with trust reference
            if result.source == trust_ref:
                weight = 1.2  # Slight boost for trust reference
            else:
                # Weight based on how much it agrees with the consensus
                consensus = 1.0 - agreement_metrics["std_dev"]
                weight = 0.8 + 0.4 * consensus
            
            weights.append(weight)
        
        # Normalize weights to sum to 1
        total = sum(weights)
        return [w/total for w in weights] if total > 0 else [1/len(weights)] * len(weights)
    
    def _update_historical_data(self, dimension: str, 
                              agreement_metrics: Dict, cluster_info: Dict):
        """Track historical agreement patterns for trend analysis"""
        if dimension not in self.historical_agreement:
            self.historical_agreement[dimension] = {
                "agreement_scores": [],
                "std_devs": [],
                "cluster_counts": []
            }
        
        hist = self.historical_agreement[dimension]
        hist["agreement_scores"].append(agreement_metrics["agreement_score"])
        hist["std_devs"].append(agreement_metrics["std_dev"])
        hist["cluster_counts"].append(cluster_info["cluster_count"])
        
        # Keep only last 1000 entries to prevent memory bloat
        if len(hist["agreement_scores"]) > 1000:
            hist["agreement_scores"] = hist["agreement_scores"][-1000:]
            hist["std_devs"] = hist["std_devs"][-1000:]
            hist["cluster_counts"] = hist["cluster_counts"][-1000:]
    
    def _generate_explanation(self, agreement_metrics: Dict, 
                            cluster_info: Dict, trust_ref: str, 
                            uncertainty: float) -> str:
        """Generate human-readable explanation of MARS analysis"""
        parts = []
        
        # Agreement assessment
        if agreement_metrics["agreement_score"] > 0.8:
            parts.append("High agreement among scorers")
        elif agreement_metrics["agreement_score"] > 0.6:
            parts.append("Moderate agreement among scorers")
        else:
            parts.append("Low agreement among scorers")
        
        # Cluster analysis
        if cluster_info["cluster_count"] > 1:
            parts.append(f"{cluster_info['cluster_count']} distinct viewpoints detected")
        
        # Uncertainty assessment
        if uncertainty > 0.5:
            parts.append("High uncertainty in final score")
        elif uncertainty > 0.3:
            parts.append("Moderate uncertainty in final score")
        
        # Trust reference
        parts.append(f"Most aligned with {trust_ref} as reference")
        
        return ". ".join(parts) + "."