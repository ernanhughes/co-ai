# stephanie/analysis/policy_analyzer.py
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from stephanie.models.evaluation_attribute import EvaluationAttributeORM
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from typing import Dict, List, Any

class PolicyAnalyzer:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.uncertainty_threshold = 0.3  # From config

    def analyze_dimension(self, dimension: str) -> Dict[str, Any]:
        """Analyze policy behavior for a specific dimension"""
        try:
            # Get policy data from database
            sicql_data = self._get_sicql_data(dimension)
            mrq_data = self._get_mrq_data(dimension)
            
            # Process and compare
            results = {
                "dimension": dimension,
                "policy_stats": self._analyze_policy_patterns(sicql_data),
                "comparison": self._compare_with_mrq(sicql_data, mrq_data),
                "uncertainty_cases": self._find_high_uncertainty(sicql_data),
                "policy_entropy": self._calculate_entropy(sicql_data),
                "policy_drift": self._detect_policy_drift(sicql_data)
            }
            
            # Log analysis
            self.logger.log("PolicyAnalysis", results)
            return results
            
        except Exception as e:
            self.logger.log("PolicyAnalysisFailed", {"error": str(e)})
            raise

    def _get_sicql_data(self, dimension: str) -> List[Dict]:
        """Get SICQL policy data from database"""
        query = (
            self.session.query(EvaluationAttributeORM)
            .join(EvaluationORM)
            .filter(
                EvaluationAttributeORM.dimension == dimension,
                EvaluationORM.evaluator_name == "sicql"
            )
        )
        
        return [self._format_attribute(attr) for attr in query.all()]

    def _get_mrq_data(self, dimension: str) -> List[Dict]:
        """Get MRQ scores for comparison"""
        query = (
            self.session.query(ScoreORM)
            .join(EvaluationORM)
            .filter(
                ScoreORM.dimension == dimension,
                EvaluationORM.evaluator_name == "mrq"
            )
        )
        
        return [self._format_score(score) for score in query.all()]

    def _format_attribute(self, attr: EvaluationAttributeORM) -> Dict:
        """Convert ORM to analysis-friendly format"""
        return {
            "evaluation_id": attr.evaluation_id,
            "policy_logits": np.array(attr.policy_logits) if attr.policy_logits else None,
            "q_value": attr.q_value,
            "v_value": attr.v_value,
            "uncertainty": attr.uncertainty,
            "advantage": attr.advantage,
            "dimension": attr.dimension,
            "timestamp": attr.created_at
        }

    def _format_score(self, score: ScoreORM) -> Dict:
        """Convert MRQ scores to comparable format"""
        return {
            "evaluation_id": score.evaluation_id,
            "score": score.score,
            "dimension": score.dimension,
            "source": score.source
        }

    def _analyze_policy_patterns(self, sicql_data: List[Dict]) -> Dict[str, Any]:
        """Analyze policy patterns across samples"""
        if not sicql_data or not any(d["policy_logits"] is not None for d in sicql_data):
            return {"available": False}
            
        # Calculate action probabilities
        all_logits = np.stack([d["policy_logits"] for d in sicql_data if d["policy_logits"] is not None])
        action_probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
        
        # Calculate entropy
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1)
        
        # Get most probable actions
        most_probable_actions = np.argmax(action_probs, axis=1)
        
        # Action distribution
        action_counts = np.bincount(most_probable_actions, minlength=action_probs.shape[1])
        
        return {
            "available": True,
            "action_distribution": action_counts.tolist(),
            "avg_entropy": float(np.mean(entropy)),
            "action_probabilities": action_probs.tolist(),
            "policy_consistency": self._calculate_policy_consistency(sicql_data)
        }

    def _calculate_policy_consistency(self, sicql_data: List[Dict]) -> Dict[str, Any]:
        """Measure policy consistency over time"""
        if len(sicql_data) < 2:
            return {"consistent": None}
            
        # Compare consecutive policy outputs
        actions = np.argmax([d["policy_logits"] for d in sicql_data if d["policy_logits"] is not None], axis=1)
        policy_changes = np.sum(actions[1:] != actions[:-1]) / len(actions) if len(actions) > 1 else 0
        
        # Compare Q/V alignment
        q_values = np.array([d["q_value"] for d in sicql_data if d["q_value"] is not None])
        v_values = np.array([d["v_value"] for d in sicql_data if d["v_value"] is not None])
        
        if len(q_values) > 1:
            q_v_correlation = np.corrcoef(q_values, v_values)[0,1]
        else:
            q_v_correlation = None
            
        return {
            "policy_stability": 1 - policy_changes,
            "q_v_correlation": float(q_v_correlation) if q_v_correlation is not None else None,
            "sample_count": len(sicql_data)
        }

    def _compare_with_mrq(self, sicql_data: List[Dict], mrq_data: List[Dict]) -> Dict[str, Any]:
        """Compare SICQL policy with MRQ scores"""
        if not sicql_data or not mrq_data:
            return {
                "comparable": False,
                "score_correlation": None,
                "avg_score_deviation": None,
                "sample_count": 0
            }

        # Match by evaluation_id
        sicql_by_id = {d["evaluation_id"]: d for d in sicql_data}
        mrq_by_id = {d["evaluation_id"]: d for d in mrq_data}
        
        matched = []
        for eid, sicql in sicql_by_id.items():
            if eid in mrq_by_id:
                matched.append({
                    "sicql": sicql,
                    "mrq": mrq_by_id[eid]
                })
                
        if not matched:
            return {
                "comparable": False,
                "score_correlation": None,
                "avg_score_deviation": None,
                "sample_count": 0
            }
            
        # Calculate score correlations
        sicql_scores = [m["sicql"]["q_value"] for m in matched]
        mrq_scores = [m["mrq"]["score"] for m in matched]
        
        try:
            score_correlation = np.corrcoef(sicql_scores, mrq_scores)[0,1]
        except:
            score_correlation = None
            
        # Compare uncertainty vs score deviation
        uncertainty_scores = [abs(m["sicql"]["q_value"] - m["mrq"]["score"]) for m in matched]
        avg_deviation = np.mean(uncertainty_scores) if uncertainty_scores else None
        
        return {
            "comparable": True,
            "score_correlation": float(score_correlation) if score_correlation is not None else None,
            "avg_score_deviation": float(avg_deviation) if avg_deviation is not None else None,
            "sample_count": len(matched)
        }

    def _find_high_uncertainty(self, sicql_data: List[Dict]) -> List[Dict]:
        """Find cases where policy was uncertain"""
        return [
            d for d in sicql_data 
            if d["uncertainty"] and d["uncertainty"] > self.uncertainty_threshold
        ]

    def _calculate_entropy(self, sicql_data: List[Dict]) -> Dict[str, float]:
        """Calculate entropy statistics"""
        valid_logits = [d["policy_logits"] for d in sicql_data if d["policy_logits"] is not None]
        if not valid_logits:
            return {}
            
        action_probs = np.exp(valid_logits) / np.exp(valid_logits).sum(axis=1, keepdims=True)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1)
        
        return {
            "avg_entropy": float(np.mean(entropy)),
            "std_entropy": float(np.std(entropy)),
            "high_entropy": int(np.sum(entropy > np.median(entropy) + np.std(entropy)))
        }

    def _detect_policy_drift(self, sicql_data: List[Dict]) -> Dict[str, Any]:
        """Detect policy drift over time"""
        if len(sicql_data) < 2:
            return {"drift_detected": False}
            
        # Sort by timestamp
        sorted_data = sorted(sicql_data, key=lambda x: x["timestamp"])
        actions = np.argmax([d["policy_logits"] for d in sorted_data if d["policy_logits"] is not None], axis=1)
        
        if len(actions) < 2:
            return {"drift_detected": False}
            
        # Calculate action changes over time
        changes = np.sum(actions[1:] != actions[:-1]) / len(actions)
        
        # Detect clusters of similar policy outputs
        from sklearn.cluster import KMeans
        action_probs = np.exp(actions) / np.exp(actions).sum()
        
        try:
            kmeans = KMeans(n_clusters=3)
            clusters = kmeans.fit_predict(action_probs.reshape(-1, 1))
            cluster_changes = np.sum(clusters[1:] != clusters[:-1]) / len(clusters)
        except:
            cluster_changes = 0
            
        return {
            "drift_detected": changes > 0.3,
            "action_drift_rate": float(changes),
            "cluster_drift_rate": float(cluster_changes),
            "sample_count": len(actions)
        }

    def generate_policy_report(self, dimension: str) -> Dict[str, Any]:
        """Generate comprehensive policy report with flattened structure"""
        analysis = self.analyze_dimension(dimension)
        
        # Extract policy stats
        policy_stats = analysis.get("policy_stats", {})
        policy_consistency = policy_stats.get("policy_consistency", {})
        
        # Extract comparison stats
        comparison = analysis.get("comparison", {})
        
        # Extract other stats
        uncertainty_cases = analysis.get("uncertainty_cases", [])
        policy_entropy = analysis.get("policy_entropy", {})
        
        report = {
            "dimension": dimension,
            "policy_available": policy_stats.get("available", False),
            "policy_stability": policy_consistency.get("policy_stability", None),
            "q_v_correlation": policy_consistency.get("q_v_correlation", None),
            "policy_entropy_avg": policy_entropy.get("avg_entropy", None),
            "policy_entropy_std": policy_entropy.get("std_entropy", None),
            "score_correlation": comparison.get("score_correlation", None),
            "score_deviation": comparison.get("avg_score_deviation", None),
            "sample_count": comparison.get("sample_count", 0),
            "uncertainty_count": len(uncertainty_cases),
            "policy_drift": policy_consistency.get("drift_detected", False),
            "action_drift_rate": policy_consistency.get("action_drift_rate", 0.0),
            "insights": self._generate_insights(analysis)
        }
        
        return report

    def _generate_insights(self, report: Dict) -> List[str]:
        """Generate actionable insights from flattened policy report"""
        insights = []
        
        # 1. Uncertainty detection
        uncertainty_count = report.get("uncertainty_count", 0)
        sample_count = report.get("sample_count", 1)  # Avoid division by 0
        uncertainty_ratio = uncertainty_count / sample_count
        
        if uncertainty_count > 0:
            insights.append(
                f"Found {uncertainty_count} high-uncertainty cases ({uncertainty_ratio:.1%} of samples). "
                "Consider retraining for this dimension."
            )
        
        # 2. Policy stability check
        policy_stability = report.get("policy_stability")
        if policy_stability is not None:
            if policy_stability < 0.7:
                insights.append(
                    f"Policy shows instability (stability: {policy_stability:.2f}). "
                    "Consider policy smoothing or additional training data."
                )
            elif policy_stability < 0.9:
                insights.append(
                    f"Moderate policy stability ({policy_stability:.2f}) - watch for drift."
                )
        
        # 3. Score alignment analysis
        score_correlation = report.get("score_correlation")
        score_deviation = report.get("score_deviation")
        
        if score_correlation is not None and score_deviation is not None:
            if score_correlation < 0.5:
                insights.append(
                    f"Low correlation ({score_correlation:.2f}) between SICQL and MRQ scores. "
                    "Policy may be misaligned with expected scoring patterns."
                )
            if score_deviation > 5.0:
                insights.append(
                    f"Large score deviation ({score_deviation:.1f}) from MRQ baseline. "
                    "Consider recalibration or investigating outlier cases."
                )
        
        # 4. Entropy analysis
        policy_entropy_avg = report.get("policy_entropy_avg")
        if policy_entropy_avg is not None:
            if policy_entropy_avg > 2.0:
                insights.append(
                    f"High policy entropy ({policy_entropy_avg:.2f}) - system is exploring broadly. "
                    "Consider focusing training on high-reward regions."
                )
            elif policy_entropy_avg < 0.5:
                insights.append(
                    f"Low policy entropy ({policy_entropy_avg:.2f}) - system is exploiting known strategies. "
                    "Consider adding diversity to training data."
                )
        
        # 5. Policy drift detection
        policy_drift = report.get("policy_drift", False)
        action_drift_rate = report.get("action_drift_rate", 0.0)
        
        if policy_drift and action_drift_rate > 0.2:
            insights.append(
                f"Policy drift detected (drift rate: {action_drift_rate:.2f}). "
                "Consider retraining or policy regularization."
            )
        
        return insights
        
    def visualize_policy(self, dimension: str, output_path: str = "policy_visualization"):
        """Generate policy visualization for a dimension"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sicql_data = self._get_sicql_data(dimension)
            if not sicql_data or not any(d["policy_logits"] for d in sicql_data):
                return None
                
            # Create DataFrame
            df = pd.DataFrame({
                "timestamp": [d["timestamp"] for d in sicql_data],
                "q_value": [d["q_value"] for d in sicql_data],
                "uncertainty": [d["uncertainty"] for d in sicql_data],
                "action": np.argmax(
                    [d["policy_logits"] for d in sicql_data if d["policy_logits"] is not None], 
                    axis=1
                ).tolist()
            })
            
            # Policy visualization
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x="timestamp", y="q_value", label="Q-value")
            sns.lineplot(data=df, x="timestamp", y="uncertainty", label="Uncertainty")
            
            # Highlight high-uncertainty periods
            high_uncertainty = df[df["uncertainty"] > self.uncertainty_threshold]
            for _, row in high_uncertainty.iterrows():
                plt.axvspan(row["timestamp"], row["timestamp"], alpha=0.2, color='red')
                
            plt.title(f"Policy Behavior for {dimension.capitalize()}")
            plt.xlabel("Time")
            plt.ylabel("Score")
            plt.legend()
            plt.savefig(f"{output_path}_behavior.png")
            plt.close()
            
            # Action distribution
            plt.figure(figsize=(8, 6))
            sns.histplot(df["action"], bins=np.arange(-0.5, 3.5, 1), discrete=True)
            plt.title(f"Action Distribution for {dimension.capitalize()}")
            plt.xlabel("Selected Action")
            plt.ylabel("Frequency")
            plt.savefig(f"{output_path}_distribution.png")
            plt.close()
            
            return {
                "behavior_plot": f"{output_path}_behavior.png",
                "distribution_plot": f"{output_path}_distribution.png"
            }
            
        except Exception as e:
            self.logger.log("PolicyVisualizationFailed", {"error": str(e)})
            return None
    
    def _generate_visualization_guidance(self, report):
        if report["policy_drift"]:
            return "policy_drift"
        if report["uncertainty_count"] > 20:
            return "uncertainty_analysis"
        return "standard_view"
    
    def _generate_insights(self, report):
        # Add retraining 
        if report.get("uncertainty_count", 0) > 30:
            yield {
                "action": "retrain",
                "reason": "High uncertainty in policy decisions",
                "urgency": 2
            }
        # Add policy smoothing recommendations
        if report.get("policy_stability", 1) < 0.7:
            yield {
                "action": "policy_smoothing",
                "reason": "Policy instability detected",
                "urgency": 1
            }

