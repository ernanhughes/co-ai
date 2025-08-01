from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.contrastive_ranker_scorer import ContrastiveRankerScorer
from stephanie.scoring.ebt_scorer import EBTScorer
from stephanie.scoring.hrm_scorer import HRMScorer
from stephanie.scoring.mrq_scorer import MRQScorer
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.svm_scorer import SVMScorer
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.data.score_result import ScoreResult
from typing import Dict, List, Any
import time
import random
from tqdm import tqdm


class DocumentRewardScorerAgent(BaseAgent):
    """
    Scores document sections or full documents to assess reward value
    using configured reward model (e.g., SVM-based or regression-based).
    
    Enhanced with MARS (Model Agreement and Reasoning Signal) analysis
    to evaluate consistency across scoring models.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", ["helpfulness", "truthfulness", "reasoning_quality"])
        self.include_mars = cfg.get("include_mars", True)
        self.mars_calculator = MARSCalculator(cfg.get("dimension_config", {}))
        self.test_mode = cfg.get("test_mode", False)
        self.test_document_count = cfg.get("test_document_count", 100)
        
        # Configure which scorers to use
        self.scorer_types = cfg.get("scorer_types", [
            "svm", "mrq", "sicql", "ebt", "hrm", "contrastive_ranker"
        ])
        
        self.scorer_types = ["sicql"]

        # Initialize scorers dynamically
        self.scorers = self._initialize_scorers()
        
        self.logger.log("DocumentRewardScorerInitialized", {
            "dimensions": self.dimensions,
            "scorers": self.scorer_types,
            "include_mars": self.include_mars,
            "test_mode": self.test_mode
        })

    def _initialize_scorers(self) -> Dict[str, Any]:
        """Initialize all configured scorers"""
        scorers = {}
        
        if "svm" in self.scorer_types:
            scorers["svm"] = SVMScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "mrq" in self.scorer_types:
            scorers["mrq"] = MRQScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "sicql" in self.scorer_types:
            scorers["sicql"] = SICQLScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "ebt" in self.scorer_types:
            scorers["ebt"] = EBTScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "hrm" in self.scorer_types:
            scorers["hrm"] = HRMScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "contrastive_ranker" in self.scorer_types:
            scorers["contrastive_ranker"] = ContrastiveRankerScorer(
                self.cfg, memory=self.memory, logger=self.logger
            )
            
        return scorers

    async def run(self, context: dict) -> dict:
        """Main execution method with optional test mode"""
        start_time = time.time()
        
        # Handle test mode if enabled
        if self.test_mode:
            documents = self._generate_test_documents()
            self.logger.log("TestModeActivated", {
                "document_count": len(documents),
                "dimensions": self.dimensions
            })
        else:
            documents = context.get(self.input_key, [])
            
        if not documents:
            self.logger.log("NoDocumentsFound", {"source": self.input_key})
            return context
            
        results = []
        mars_results = []
        total_documents = len(documents)
        
        # Process documents with progress tracking
        pbar = tqdm(
            documents, 
            desc="Scoring Documents", 
            total=total_documents,
            disable=not self.cfg.get("progress", True)
        )
        
        for idx, doc in enumerate(pbar):
            try:
                # Score document with all scorers
                scoring_start = time.time()
                doc_scores, doc_mars = self._score_document(context, doc)
                scoring_time = time.time() - scoring_start
                
                # Update progress bar
                pbar.set_postfix({
                    "docs": f"{idx+1}/{total_documents}",
                    "scorers": len(self.scorers)
                })
                
                # Log performance metrics
                if (idx + 1) % 10 == 0 or idx == total_documents - 1:
                    self.logger.log("DocumentScoringProgress", {
                        "processed": idx + 1,
                        "total": total_documents,
                        "avg_time_per_doc": scoring_time,
                        "scorers": len(self.scorers)
                    })
                
                # Store results
                results.append(doc_scores)
                if doc_mars:
                    mars_results.append(doc_mars)
                    
            except Exception as e:
                self.logger.log("DocumentScoringError", {
                    "document_id": doc.get("id", "unknown"),
                    "error": str(e)
                })
                continue
        
        # Save MARS results to context
        if mars_results and self.include_mars:
            context["mars_analysis"] = {
                "summary": self._summarize_mars_results(mars_results),
                "details": mars_results
            }
            self.logger.log("MARSAnalysisCompleted", {
                "document_count": len(mars_results),
                "dimensions": self.dimensions
            })
        
        # Save results to context
        context[self.output_key] = results
        context["scoring_time"] = time.time() - start_time
        context["total_documents"] = total_documents
        context["scorers_used"] = list(self.scorers.keys())
        
        self.logger.log("DocumentScoringComplete", {
            "total_documents": total_documents,
            "dimensions": self.dimensions,
            "scorers": len(self.scorers),
            "total_time": context["scoring_time"]
        })
        
        return context

    def _score_document(self, context: dict, doc: dict) -> tuple:
        """Score a single document with all configured scorers"""
        doc_id = doc["id"]
        goal = context.get("goal", {"goal_text": ""})
        scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
        
        # Collect scores from all scorers
        all_scores = {}
        scorer_timings = {}
        
        for scorer_name, scorer in self.scorers.items():
            try:
                start_time = time.time()
                score_bundle = scorer.score(
                    goal=goal,
                    scorable=scorable,
                    dimensions=self.dimensions,
                )
                scorer_timings[scorer_name] = time.time() - start_time
                
                # Convert to dictionary format for easier processing
                for dim, result in score_bundle.results.items():
                    if dim not in all_scores:
                        all_scores[dim] = {}
                    all_scores[dim][scorer_name] = result.score
                
                # Save individual scores to memory
                ScoringManager.save_score_to_memory(
                    score_bundle,
                    scorable,
                    context,
                    self.cfg,
                    self.memory,
                    self.logger,
                    source=scorer.model_type,
                    model_name=scorer.get_model_name(),
                )
                
            except Exception as e:
                self.logger.log("ScorerError", {
                    "scorer": scorer_name,
                    "document_id": doc_id,
                    "error": str(e)
                })
                continue
        
        # Create score bundle for reporting
        from stephanie.data.score_bundle import ScoreBundle

        score_bundle = ScoreBundle(
            results={dim: list(scores.keys())[0] for dim, scores in all_scores.items()},
            dimension_config=self.cfg.get("dimension_config", {})
        )
        
        # Generate MARS analysis if requested
        mars_result = None
        if self.include_mars and all_scores:
            mars_result = self._analyze_with_mars(all_scores, doc_id, goal)
        
        return {
            "document_id": doc_id,
            "title": doc.get("title", ""),
            "scores": all_scores,
            "scorer_timings": scorer_timings,
            "goal_text": goal.get("goal_text", "")
        }, mars_result

    def _analyze_with_mars(self, all_scores: Dict, doc_id: str, goal: dict) -> Dict:
        """Perform MARS analysis on collected scores"""
        mars_details = {}
        
        for dimension, scores in all_scores.items():
            # Create a temporary ScoreBundle for this dimension
            bundle_results = {}
            for scorer_name, score_value in scores.items():
                bundle_results[scorer_name] = ScoreResult(
                    dimension=dimension,
                    score=score_value,
                    rationale=f"Score from {scorer_name}",
                    source=scorer_name,
                    weight=1.0
                )
            from stephanie.data.score_bundle import ScoreBundle

            bundle = ScoreBundle(
                results=bundle_results,
                dimension_config=self.cfg.get("dimension_config", {})
            )
            
            # Perform MARS analysis
            mars_analysis = self.mars_calculator.calculate(bundle)
            
            mars_details[dimension] = {
                "agreement": mars_analysis["agreement"],
                "score": mars_analysis["score"],
                "uncertainty": mars_analysis["uncertainty"],
                "trust_reference": mars_analysis["trust_reference"],
                "explanation": mars_analysis["explanation"]
            }
        
        return {
            "document_id": doc_id,
            "goal_text": goal.get("goal_text", ""),
            "mars_analysis": mars_details
        }

    def _summarize_mars_results(self, mars_results: List[Dict]) -> Dict:
        """Create summary statistics from multiple MARS analyses"""
        if not mars_results:
            return {}
            
        summary = {
            "dimensions": {},
            "overall_agreement": 0.0,
            "high_uncertainty_count": 0,
            "total_documents": len(mars_results)
        }
        
        # Collect data by dimension
        dimension_data = {}
        for result in mars_results:
            for dim, analysis in result["mars_analysis"].items():
                if dim not in dimension_data:
                    dimension_data[dim] = {
                        "agreement_scores": [],
                        "uncertainties": []
                    }
                dimension_data[dim]["agreement_scores"].append(analysis["agreement"]["agreement_score"])
                dimension_data[dim]["uncertainties"].append(analysis["uncertainty"])
                
                if analysis["uncertainty"] > 0.5:
                    summary["high_uncertainty_count"] += 1
        
        # Calculate summary stats
        for dim, data in dimension_data.items():
            summary["dimensions"][dim] = {
                "avg_agreement": sum(data["agreement_scores"]) / len(data["agreement_scores"]),
                "avg_uncertainty": sum(data["uncertainties"]) / len(data["uncertainties"]),
                "high_uncertainty_pct": (
                    sum(1 for u in data["uncertainties"] if u > 0.5) / len(data["uncertainties"])
                )
            }
        
        # Overall agreement is average of dimension agreements
        summary["overall_agreement"] = sum(
            stats["avg_agreement"] for stats in summary["dimensions"].values()
        ) / len(summary["dimensions"])
        
        return summary

    def _generate_test_documents(self) -> List[Dict]:
        """Generate synthetic documents for testing"""
        self.logger.log("GeneratingTestDocuments", {
            "count": self.test_document_count,
            "dimensions": self.dimensions
        })
        
        documents = []
        for i in range(self.test_document_count):
            # Generate realistic-looking content
            doc_type = random.choice(["article", "research_paper", "blog_post", "technical_doc"])
            length = random.randint(100, 2000)
            
            documents.append({
                "id": f"test_doc_{i}",
                "title": f"Test Document #{i} - {doc_type}",
                "content": " ".join([f"word_{j}" for j in range(length)]),
                "type": doc_type,
                "length": length
            })
            
        return documents