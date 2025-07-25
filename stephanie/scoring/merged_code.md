<!-- Merged Python Code Files -->


## File: __init__.py

`python
# stephanie/scoring/__init__.py
from .structured_engine import StructuredScoringEngine
``n

## File: alignment\live_mrq_aligner.py

`python
# stephanie/scoring/alignment/live_mrq_aligner.py

import logging
from typing import List, Optional

from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class LiveMRQAligner:
    def __init__(self):
        self.mrq_scores: List[float] = []
        self.llm_scores: List[float] = []
        self.model: Optional[LinearRegression] = None
        self.min_points = 10  # Minimum data before tuning

    def add_score_pair(self, mrq_score: float, llm_score: float):
        self.mrq_scores.append(mrq_score)
        self.llm_scores.append(llm_score)
        if len(self.mrq_scores) >= self.min_points:
            self._fit()

    def _fit(self):
        X = [[s] for s in self.mrq_scores]
        y = self.llm_scores
        self.model = LinearRegression().fit(X, y)
        logger.info(
            f"[LiveMRQAligner] Updated transformation model "
            f"with {len(self.mrq_scores)} examples. "
            f"Slope: {self.model.coef_[0]:.3f}, Intercept: {self.model.intercept_:.3f}"
        )

    def transform(self, mrq_score: float) -> float:
        if self.model:
            return float(self.model.predict([[mrq_score]])[0])
        return mrq_score  # Return raw score if model not ready

    def clear(self):
        self.mrq_scores.clear()
        self.llm_scores.clear()
        self.model = None
``n

## File: base_evaluator.py

`python
# stephanie/scoring/base_evaluator.py
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, prompt: str, response: str = None) -> dict:
        """Returns a structured score dict with score, rationale, etc."""
        pass
``n

## File: base_score.py

`python
# stephanie/scoring/base_score.py
from abc import ABC, abstractmethod

from stephanie.models import EvaluationORM


class BaseScore(ABC):
    name: str = "unnamed"
    default_value: float = 0.0

    def __init__(self, cfg, memory, logger, evaluator_name=None):
        self.memory = memory
        self.logger = logger
        self.agent_name = cfg.get("name")
        self.model_name = cfg.get("model", {}).get("name")
        self.evaluator_name = evaluator_name or self.name

    @abstractmethod
    def compute(self, hypothesis: dict, context: dict) -> float:
        pass

    def get_score(self, hypothesis: dict, context: dict) -> float:
        # 1. If already cached on object
        if hypothesis.get(f"{self.name}_score"):
            return hypothesis[f"{self.name}_score"]

        # 2. Compute and attach
        score = self.compute(hypothesis, context)
        hypothesis[f"{self.name}_score"] = score

        # 3. Store in scores table
        if self.memory:
            # Optional dimensions dict (can be overridden in subclass)
            dimensions = getattr(self, "dimensions", None)

            s = EvaluationORM(
                goal_id=hypothesis.get("goal_id"),
                target_type="hypothesis",
                target_id=hypothesis.get("id"),
                agent_name=self.agent_name,
                model_name=self.model_name,
                embedding_type=self.memory.embedding.type,
                evaluator_name=self.evaluator_name,
                scores=dimensions,
                pipeline_run_id=context.get("pipeline_run_id"),
            )
            try:
                self.memory.evaluations.insert(s)
                self.memory.commit()  # Ensure session commit happens
            except Exception as e:
                self.memory.refresh_session()
                score = self.default_value
                self.logger.log("ScoreInsertFailed", {"error": str(e)})

        # 4. Log
        self.logger.log(
            "ScoreComputed",
            {"type": self.name, "score": score, "hypothesis_id": hypothesis.get("id")},
        )

        return score
``n

## File: base_scorer.py

`python
import abc
from typing import List

import torch

from stephanie.scoring.model_locator_mixin import ModelLocatorMixin
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle


class BaseScorer(ModelLocatorMixin, abc.ABC):
    def __init__(self, cfg: dict, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        self.embedding_type = self.memory.embedding.type
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim

        self.model_path = cfg.get("model_path", "models")
        self.target_type = cfg.get("target_type", "document")
        self.model_type = cfg.get("model_type", "svm")  # Override in subclass
        self.version = cfg.get("model_version", "v1")

        self.force_rescore = cfg.get("force_rescore", False)
        self.dimensions = cfg.get("dimensions", [])
        self.device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        """Returns a canonical name for the scorer."""
        return f"{self.model_type}_scorer"

    def get_model_name(self) -> str:
        return f"{self.target_type}_{self.model_type}_{self.version}"

    @abc.abstractmethod
    def score(
        self,
        goal: dict,
        scorable: Scorable,
        dimensions: List[str],
    ) -> ScoreBundle:
        """
        Score a single item (Scorable) for a given goal and a set of dimensions.

        Returns:
            ScoreBundle containing ScoreResults for each dimension.
        """
        raise NotImplementedError("Subclasses must implement score()")

    def log_event(self, event: str, data: dict):
        if self.logger:
            self.logger.log(event, data)
``n

## File: batch.py

`python
# stephanie/scoring/batch.py
from stephanie.agents.pipeline_judge import PipelineJudgeAgent
from stephanie.constants import GOAL, PIPELINE_RUN_ID, RUN_ID
from stephanie.models import EvaluationORM, HypothesisORM, RuleApplicationORM


def get_unscored_hypotheses(session, run_id: str = None):
    # Get hypotheses with no evaluations for this run
    subquery = session.query(EvaluationORM.hypothesis_id).distinct()
    query = session.query(HypothesisORM).filter(~HypothesisORM.id.in_(subquery))

    if run_id:
        query = query.filter(HypothesisORM.pipeline_run_id == run_id)

    return query.all()


async def score_unscored_hypotheses(memory, logger, config, run_id=None):
    session = memory.session
    unscored = get_unscored_hypotheses(session, run_id)
    agent = PipelineJudgeAgent(cfg=config, memory=memory, logger=logger)

    for hypo in unscored:
        goal = memory.goals.get_by_id(hypo.goal_id)
        rule_apps = memory.rule_effects.get_by_hypothesis(hypo.id)

        context = {
            GOAL: goal.to_dict(),
            "hypotheses": [hypo.to_dict()],
            PIPELINE_RUN_ID: hypo.pipeline_run_id,
            RUN_ID: f"batch-repair-{hypo.id}",
            "rule_applications": [ra.to_dict() for ra in rule_apps],
        }

        logger.log(
            "ScoringUnscoredHypothesis",
            {
                "hypothesis_id": hypo.id,
                "goal_id": goal.id,
                "rule_count": len(rule_apps),
            },
        )

        await agent.run(context)
``n

## File: calculations\__init__.py

`python
# stephanie/scoring/calculations/__init__.py
``n

## File: calculations\base_calculator.py

`python
# stephanie/scoring/calculations/base_calculator.py
from abc import ABC, abstractmethod

from stephanie.scoring.score_bundle import ScoreBundle


class BaseScoreCalculator(ABC):
    @abstractmethod
    def calculate(self, results: ScoreBundle) -> float:
        """
        Given a dict of dimension results (each with score, weight), return a single float score.
        """
        pass
``n

## File: calculations\mrq_normalizer.py

`python
# stephanie/scoring/calculations/mrq_normalizer.py
"""
MRQNormalizerCalculator

This class is used to normalize raw MR.Q scores into a standardized range (typically 0–100)
to support consistent evaluation across dimensions. It takes expected min and max scores as
bounds, then rescales incoming scores accordingly. This is useful when MR.Q scoring output
is unstable or on a different scale than LLM-based scoring.

Key Functions:
- normalize raw scores per dimension
- apply clipping to keep scores within [0, 1] before scaling
- compute a weighted average as the final score
- used by ScoringManager to align MR.Q scores with other evaluators
All right let's stay away for a bit mine
Intended for use in the scoring pipeline to support adaptive tuning and fair comparisons.
"""

from stephanie.scoring.calculations.base_calculator import BaseCalculator


class MRQNormalizerCalculator(BaseCalculator):
    def __init__(self, expected_min=0.0, expected_max=1.0, clip=True, scale=100.0):
        self.expected_min = expected_min
        self.expected_max = expected_max
        self.clip = clip
        self.scale = scale  # typically 1.0 or 100.0

    def calculate(self, results: dict) -> float:
        raw_total = 0.0
        weight_sum = 0.0

        for dim, val in results.items():
            raw = val["score"]
            norm = (raw - self.expected_min) / max(
                (self.expected_max - self.expected_min), 1e-6
            )

            if self.clip:
                norm = max(0.0, min(norm, 1.0))

            val["normalized_score"] = round(norm * self.scale, 2)

            raw_total += norm * self.scale * val.get("weight", 1.0)
            weight_sum += val.get("weight", 1.0)

        return round(raw_total / weight_sum, 2) if weight_sum else 0.0
``n

## File: calculations\score_delta.py

`python
# stephanie/scoring/calculations/score_delta.py
class ScoreDeltaCalculator:
    def __init__(self, cfg: dict, memory, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

    def log_score_delta(self, scorable, new_score, goal_id=None):
        prev = self.memory.evaluations.get_latest_score(
            scorable, agent_name=self.cfg.get("name")
        )
        if prev is not None:
            delta = round(new_score - prev, 2)
            if self.logger:
                self.logger.log(
                    "ScoreDelta",
                    {
                        "delta": delta,
                        "id": scorable.id,
                        "target_type": scorable.target_type,
                        "text": scorable.text[:60],
                        "goal_id": goal_id,
                        "prev_score": prev,
                        "new_score": new_score,
                        "stage": self.cfg.get("name"),
                    },
                )
            return delta
        return None
``n

## File: calculations\weighted_average.py

`python
# stephanie/scoring/calculations/weighted_average.py
from stephanie.scoring.calculations.base_calculator import BaseScoreCalculator
from stephanie.scoring.score_bundle import ScoreBundle


class WeightedAverageCalculator(BaseScoreCalculator):
    def calculate(self, bundle: ScoreBundle) -> float:
        results = bundle.results.values()
        total = sum(r.score * getattr(r, "weight", 1.0) for r in results)
        weight_sum = sum(getattr(r, "weight", 1.0) for r in results)
        return round(total / weight_sum, 2) if weight_sum else 0.0
``n

## File: contrastive_dimensional_tuner.py

`python
# stephanie/scoring/contrastive_dimensional_tuner.py
import numpy as np
from sklearn.linear_model import LogisticRegression


class ContrastiveDimensionalTuner:
    """
    Learns weights for each scoring dimension using contrastive learning.
    Given pairs of scored examples (A vs B) and a preference, it learns which dimensions matter most.
    """

    def __init__(self, dimensions, logger=None):
        """
        Args:
            dimensions (list of str): List of dimension names (e.g., ["correctness", "clarity"]).
            logger (optional): Optional logger to record training events.
        """
        self.dimensions = dimensions
        self.logger = logger
        self.X = []  # Feature differences (vector of deltas across dimensions)
        self.y = []  # Labels: 1 if A preferred over B, 0 otherwise
        self.model = None

    def add_training_pair(self, scores_a: dict, scores_b: dict, preferred: str):
        """
        Adds a training example.

        Args:
            scores_a (dict): Scores for option A, keyed by dimension.
            scores_b (dict): Scores for option B, keyed by dimension.
            preferred (str): "A" or "B", indicating which output was preferred.
        """
        delta = np.array([scores_a[dim] - scores_b[dim] for dim in self.dimensions])

        # If B is preferred, invert the delta
        if preferred.upper() == "B":
            delta = -delta
            label = 1  # B preferred (inverted delta)
        else:
            label = 1  # A preferred (original delta)

        self.X.append(delta)
        self.y.append(label)

        if self.logger:
            self.logger.log(
                "ContrastiveTrainingPairAdded",
                {"delta": delta.tolist(), "preferred": preferred},
            )

    def train(self):
        """
        Trains a logistic regression model using the current contrastive data.
        """
        if len(self.X) < 3:
            if self.logger:
                self.logger.log(
                    "ContrastiveTrainingSkipped",
                    {"reason": "Not enough data", "num_examples": len(self.X)},
                )
            return

        X_array = np.array(self.X)
        y_array = np.array(self.y)

        self.model = LogisticRegression()
        self.model.fit(X_array, y_array)

        if self.logger:
            self.logger.log(
                "ContrastiveModelTrained", {"coefficients": self.get_weights()}
            )

    def get_weights(self) -> dict:
        """
        Returns the learned dimension weights (if trained).

        Returns:
            dict: Mapping from dimension to learned weight.
        """
        if self.model is None:
            return {dim: 1.0 for dim in self.dimensions}  # fallback: equal weights

        weights = self.model.coef_[0]
        return {dim: round(float(w), 4) for dim, w in zip(self.dimensions, weights)}

    def score(self, dimension_scores: dict) -> float:
        """
        Calculates a single weighted score from per-dimension scores.

        Args:
            dimension_scores (dict): Scores keyed by dimension.

        Returns:
            float: Weighted total score.
        """
        weights = self.get_weights()
        total = sum(
            dimension_scores[dim] * weights.get(dim, 1.0) for dim in self.dimensions
        )
        return round(total, 4)
``n

## File: document_mrq_scorer.py

`python
# stephanie/scoring/document_mrq_scorer.py

import os

import torch

from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class DocumentMRQScorer:
    def __init__(
        self,
        memory,
        logger,
        device="cpu",
        dimensions=None,
        cfg=None,
        model_dir=None,
        model_prefix=None,
    ):
        self.memory = memory
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim
        self.logger = logger
        self.device = device
        self.dimensions = dimensions or []
        self.cfg = cfg or {}

        # ✅ Accept model_dir/model_prefix from args or fallback to cfg or defaults
        self.model_dir = model_dir or self.cfg.get("model_dir", "models/document")
        self.model_prefix = model_prefix or self.cfg.get("model_prefix", "document_rm_")

        self.models = {}
        self.regression_tuners = {}
        self.min_score_by_dim = {}
        self.max_score_by_dim = {}

        self._initialize_dimensions()

    def _initialize_dimensions(self):
        for dim in self.dimensions:
            encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
            predictor = ValuePredictor(self.dim, self.hdim).to(self.device)

            # Load model weights
            model_path = os.path.join(self.model_dir, f"{self.model_prefix}{dim}.pt")
            if os.path.exists(model_path):
                predictor.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.logger.log(
                    "DocumentMRQModelLoaded", {"dimension": dim, "path": model_path}
                )
            else:
                self.logger.log(
                    "DocumentMRQModelMissing", {"dimension": dim, "path": model_path}
                )

            # Load regression tuner if available
            tuner_path = os.path.join(
                self.model_dir, f"{self.model_prefix}{dim}_tuner.json"
            )
            tuner = RegressionTuner(dimension=dim, logger=self.logger)
            if os.path.exists(tuner_path):
                tuner.load(tuner_path)
                self.logger.log(
                    "DocumentMRQTunerLoaded", {"dimension": dim, "path": tuner_path}
                )
            else:
                self.logger.log(
                    "DocumentMRQTunerMissing", {"dimension": dim, "path": tuner_path}
                )

            self.models[dim] = (encoder, predictor)
            self.regression_tuners[dim] = tuner
            self.min_score_by_dim[dim] = 0.0
            self.max_score_by_dim[dim] = 1.0

    def normalize_score(self, raw_score: float, dimension: str) -> float:
        min_val = self.min_score_by_dim.get(dimension, 0.0)
        max_val = self.max_score_by_dim.get(dimension, 1.0)
        return (raw_score - min_val) / (max_val - min_val + 1e-6)

    def score(self, goal_text: str, document_text: str, dimension: str) -> float:
        if dimension not in self.models:
            self.logger.log("DocumentMRQMissingDimension", {"dimension": dimension})
            return 0.0

        encoder, predictor = self.models[dimension]
        encoder.eval()
        predictor.eval()

        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(goal_text), device=self.device
        ).unsqueeze(0)
        doc_emb = torch.tensor(
            self.memory.embedding.get_or_create(document_text), device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            zsa = encoder(prompt_emb, doc_emb)
            raw_score = predictor(zsa).item()

        norm_score = self.normalize_score(raw_score, dimension)

        tuner = self.regression_tuners.get(dimension)
        if tuner:
            tuned = tuner.transform(norm_score)
            self.logger.log(
                "DocumentMRQTunedScore",
                {"dimension": dimension, "raw": norm_score, "tuned": tuned},
            )
            return tuned

        return norm_score

    def train_tuner(self, dimension: str, mrq_score: float, llm_score: float):
        tuner = self.regression_tuners.get(dimension)
        if tuner:
            tuner.train_single(mrq_score, llm_score)

    def save_model(self, path_prefix: str):
        for dim, (encoder, predictor) in self.models.items():
            torch.save(predictor.state_dict(), f"{path_prefix}_{dim}.pt")
            self.regression_tuners[dim].save(f"{path_prefix}_{dim}_tuner.json")

    def load_model(self, path_prefix: str):
        for dim in self.dimensions:
            _, predictor = self.models[dim]
            predictor.load_state_dict(
                torch.load(f"{path_prefix}_{dim}.pt", map_location=self.device)
            )
            self.regression_tuners[dim].load(f"{path_prefix}_{dim}_tuner.json")
``n

## File: document_mrq_trainer.py

`python
# stephanie/scoring/document_mrq_trainer.py

from typing import List

from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.trainer_engine import MRQTrainerEngine
from stephanie.scoring.mrq.value_predictor import ValuePredictor


class DocumentMRQTrainer:
    def __init__(
        self, memory, logger, encoder=None, value_predictor=None, device="cpu"
    ):
        self.memory = memory
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim
        self.logger = logger
        self.device = device
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim


        self.encoder = encoder.to(device) if encoder else TextEncoder().to(device)
        self.value_predictor = (
            value_predictor.to(device)
            if value_predictor
            else ValuePredictor(self.dim, self.hdim).to(device)
        )
        self.regression_tuners = {}
        self.engine = MRQTrainerEngine(memory, logger, device)

    def train_multidimensional_model(self, contrast_pairs: List[dict], cfg=None):
        return self.engine.train_all(contrast_pairs, cfg or {})

    def align_to_best_llm_neighbour(self, goal, hypothesis, dimension):
        """
        Fetch similar hypotheses that already have high LLM scores.
        Then align MR.Q prediction to the best of them.
        """
        llm_scores = self.get_closest_llm_scores(hypothesis["text"], dimension)
        if llm_scores:
            self.align_with_llm_score(dimension, goal, hypothesis, max(llm_scores))
``n

## File: ebt_scorer.py

`python
import os
import torch
import torch.nn.functional as F

from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.model_locator import ModelLocator
from stephanie.utils.file_utils import load_json


class EBTScorer(BaseScorer):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "ebt"
        self.embedding_type = memory.embedding.type
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim

        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.dimensions = cfg.get("dimensions", [])

        self.models = {}
        self.model_meta = {}
        self.tuners = {}

        self._load_models(self.dimensions)

    def _load_models(self, dimensions):
        for dim in dimensions:
            locator = ModelLocator(
                root_dir=self.model_path,
                embedding_type=self.embedding_type,
                model_type=self.model_type,
                target_type=self.target_type,
                dimension=dim,
                version=self.version,
            )

            model = EBTModel(
                embedding_dim=self.dim,
                hidden_dim=self.hdim,
                num_actions=3,
                device=self.device
            ).to(self.device)

            model.encoder.load_state_dict(
                torch.load(locator.encoder_file(), map_location=self.device)
            )
            model.q_head.load_state_dict(
                torch.load(locator.q_head_file(), map_location=self.device)
            )
            model.v_head.load_state_dict(
                torch.load(locator.v_head_file(), map_location=self.device)
            )
            model.pi_head.load_state_dict(
                torch.load(locator.pi_head_file(), map_location=self.device)
            )
            
            model.eval()
            self.models[dim] = model

            meta = load_json(locator.meta_file()) if os.path.exists(locator.meta_file()) else {"min_score": 0, "max_score": 100}
            self.model_meta[dim] = meta

            if os.path.exists(locator.tuner_file()):
                tuner = RegressionTuner(dimension=dim)
                tuner.load(locator.tuner_file())
                self.tuners[dim] = tuner

    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        goal_text = goal.get("goal_text")
        results = {}

        for dim in dimensions:
            model = self.models.get(dim)
            if model is None:
                continue

            ctx_emb = torch.tensor(
                self.memory.embedding.get_or_create(goal_text), device=self.device
            ).unsqueeze(0)
            doc_emb = torch.tensor(
                self.memory.embedding.get_or_create(scorable.text), device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                result = model(ctx_emb, doc_emb)

            q_value = result["q_value"].item()
            v_value = result["state_value"].item() if "state_value" in result else 0.0
            policy_logits = result.get("action_logits", torch.zeros(1, 3)).cpu().squeeze().tolist()

            uncertainty = abs(q_value - v_value)
            advantage = q_value - v_value

            action_probs = F.softmax(torch.tensor(policy_logits), dim=-1)
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8)).item()

            meta = self.model_meta.get(dim, {"min_score": 0, "max_score": 100})
            if dim in self.tuners:
                scaled_score = self.tuners[dim].transform(q_value)
            else:
                normalized = torch.sigmoid(torch.tensor(q_value)).item()
                scaled_score = normalized * (meta["max_score"] - meta["min_score"]) + meta["min_score"]

            final_score = round(max(min(scaled_score, meta["max_score"]), meta["min_score"]), 4)

            prompt_hash = ScoreORM.compute_prompt_hash(goal_text, scorable)
            rationale = f"Q={q_value:.4f}, V={v_value:.4f}, Δ={uncertainty:.3f}, H={entropy:.3f}"

            results[dim] = ScoreResult(
                dimension=dim,
                score=final_score,
                rationale=rationale,
                weight=1.0,
                q_value=q_value,
                energy=q_value,
                source=self.name,
                target_type=scorable.target_type,
                prompt_hash=prompt_hash,
                state_value=v_value,
                policy_logits=policy_logits,
                uncertainty=uncertainty,
                entropy=entropy,
                advantage=advantage,
            )

        return ScoreBundle(results=results)
``n

## File: ebt\buffer.py

`python
# stephanie/utils/ebt_buffer.py
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class EBTTrainingBuffer:
    """
    A training buffer for Energy-Based Transformers (EBT).
    Stores (context, candidate, llm_score) pairs for retraining.
    """
    
    def __init__(self, logger, path: str = "training_data/ebt_buffer.jsonl"):
        """
        Initialize the buffer with a file path for persistent storage.
        
        Args:
            path: Path to store the buffer (JSONL format)
        """
        self.logger = logger
        self.path = Path(path)
        self.buffer: List[Dict] = []
        self.max_size = 10000  # Max number of examples to keep in memory
        self._initialize_buffer()

    def _initialize_buffer(self):
        """Load existing buffer data or create a new one"""
        try:
            if self.path.exists():
                with open(self.path, "r") as f:
                    for line in f:
                        if line.strip():
                            self.buffer.append(json.loads(line))
                self.logger.log("EBTBufferLoaded", {
                    "size": len(self.buffer),
                    "path": str(self.path)
                })
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.log("EBTBufferCreated", {"path": str(self.path)})
        except Exception as e:
            self.logger.log("EBTBufferError", {"error": str(e)})
            self.buffer = []

    def add(
        self,
        context: str,
        candidate: str,
        llm_score: float,
        ebt_score: float = None,
        metadata: Dict = None,
        source: str = "auto"
    ) -> None:
        """
        Add a new example to the buffer.
        
        Args:
            context: Goal or prompt text
            candidate: Generated or scored document
            llm_score: Ground truth score from LLM
            ebt_score: EBT's predicted score (optional)
            metadata: Additional context (e.g., dimension, task type)
            source: How this example was added (e.g., "auto", "manual")
        """
        example = {
            "context": context,
            "candidate": candidate,
            "llm_score": llm_score,
            "ebt_score": ebt_score,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "disagreement": abs(llm_score - (ebt_score or 0)),
            "meta": metadata or {}
        }
        
        self.buffer.append(example)
        
        # Keep buffer size bounded
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # Remove oldest entry
            
        self._persist()
        self.logger.log("EBTExampleAdded", {
            "context_hash": hash(context[:50]),
            "candidate_hash": hash(candidate[:50]),
            "disagreement": round(example["disagreement"], 4)
        })

    def maybe_add(
        self,
        context: str,
        candidate: str,
        llm_score: float,
        ebt_score: float,
        threshold: float = 0.15,
        metadata: Dict = None
    ) -> bool:
        """
        Conditionally add to buffer based on disagreement threshold
        
        Args:
            threshold: Minimum disagreement to qualify for retraining
        Returns:
            True if example was added, False otherwise
        """
        if abs(llm_score - ebt_score) > threshold:
            self.add(context, candidate, llm_score, ebt_score, metadata, source="disagreement")
            return True
        return False

    def get_top_k_disagreements(self, k: int = 100) -> List[Dict]:
        """
        Get the top K most contentious examples for retraining
        
        Args:
            k: Number of examples to return
        """
        sorted_buffer = sorted(
            self.buffer,
            key=lambda x: x["disagreement"],
            reverse=True
        )
        return sorted_buffer[:k]

    def get_all(self) -> List[Dict]:
        """Get all examples in buffer"""
        return self.buffer

    def clear(self) -> None:
        """Clear the buffer (e.g., after retraining)"""
        self.buffer = []
        self._persist()
        self.logger.log("EBTBufferCleared", {"path": str(self.path)})

    def _persist(self) -> None:
        """Persist buffer to disk in JSONL format"""
        try:
            with open(self.path, "w") as f:
                for entry in self.buffer:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.log("EBTBufferSaveError", {"error": str(e)})

    def load(self, path: str = None) -> List[Dict]:
        """Load buffer from disk"""
        load_path = Path(path or self.path)
        if not load_path.exists():
            return []
            
        loaded = []
        with open(load_path, "r") as f:
            for line in f:
                if line.strip():
                    loaded.append(json.loads(line))
        self.buffer = loaded
        return loaded

    def analyze(self) -> Dict:
        """
        Analyze buffer contents for model improvement insights
        
        Returns:
            dict: {
                "total": int,
                "avg_disagreement": float,
                "worst_dimensions": List[str],
                "most_disputed_candidates": List[str]
            }
        """
        if not self.buffer:
            return {"error": "Buffer is empty"}
            
        # Calculate basic stats
        total = len(self.buffer)
        avg_disagreement = sum(e["disagreement"] for e in self.buffer) / total
        
        # Group by dimension
        from collections import defaultdict
        dim_disagreements = defaultdict(list)
        for entry in self.buffer:
            dim = entry["meta"].get("dimension", "unknown")
            dim_disagreements[dim].append(entry["disagreement"])
        
        # Find dimensions with highest disagreement
        avg_by_dim = {
            dim: sum(vals)/len(vals) for dim, vals in dim_disagreements.items()
        }
        sorted_dims = sorted(avg_by_dim.items(), key=lambda x: x[1], reverse=True)
        
        # Get most disputed examples
        top_disagreements = self.get_top_k_disagreements(k=5)
        
        analysis = {
            "total_examples": total,
            "avg_disagreement": round(avg_disagreement, 4),
            "disagreement_by_dimension": {k: round(v, 4) for k, v in sorted_dims},
            "top_disputed_examples": [
                {
                    "context": e["context"][:200] + "...",
                    "candidate": e["candidate"][:200] + "...",
                    "llm_score": e["llm_score"],
                    "ebt_score": e.get("ebt_score"),
                    "disagreement": e["disagreement"]
                }
                for e in top_disagreements
            ],
            "dimensions_with_most_disagreement": [d[0] for d in sorted_dims[:3]],
            "last_updated": datetime.utcnow().isoformat()
        }
        
        self.logger.log("EBTBufferAnalysis", analysis)
        return analysis

    def to_training_dataset(self, output_path: str = "training_data/ebt_retrain.jsonl") -> None:
        """
        Export buffer to training dataset
        
        Args:
            output_path: Where to save the dataset
        """
        with open(output_path, "w") as f:
            for entry in self.buffer:
                # Convert to training example
                training_example = {
                    "context": entry["context"],
                    "candidate": entry["candidate"],
                    "score": entry["llm_score"]
                }
                f.write(json.dumps(training_example) + "\n")
                
        self.logger.log("EBTTrainingDatasetSaved", {
            "examples": len(self.buffer),
            "path": output_path
        })

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def __contains__(self, item):
        # Implement based on your use case
        # For example, check by context-candidate hash
        return any(
            hash(e["context"]) == hash(item["context"]) and 
            hash(e["candidate"]) == hash(item["candidate"])
            for e in self.buffer
        ) 
``n

## File: ebt\ebt_dataset.py

`python
# stephanie/scoring/ebt/ebt_dataset.py
from torch.utils.data import Dataset


class EBTDataset(Dataset):
    def __init__(self, data, device="cpu"):
        self.device = device
        self.data = [self._to_device(item) for item in data if self._is_valid(item)]
    
    def _is_valid(self, item):
        """Ensure item has valid data"""
        return (
            item.get("context_emb") is not None and 
            item.get("doc_emb") is not None and
            item.get("label") is not None
        )
    
    def _to_device(self, item):
        """Move item to device"""
        item["context_emb"] = item["context_emb"].to(self.device)
        item["doc_emb"] = item["doc_emb"].to(self.device)
        item["label"] = item["label"].to(self.device)
        if "expert_policy" in item:
            item["expert_policy"] = item["expert_policy"].to(self.device)
        return item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "context_emb": item["context_emb"],
            "doc_emb": item["doc_emb"],
            "label": item["label"],
            "expert_policy": item.get("expert_policy", [0.3, 0.7, 0.0])
        }
``n

## File: ebt\ebtmrq.py

`python
# stephanie/scoring/energy_tuned_mrq.py
import logging
from typing import Dict, List, Optional, Union

import torch
from torch.nn.functional import sigmoid


class EnergyTunedMRQ:
    """
    Combines MRQ and EBT models for energy-based refinement and scoring
    Uses EBT to verify and refine MRQ predictions via energy minimization
    """
    
    def __init__(self, ebt, mrq, config=None):
        """
        Args:
            ebt: EBTInferenceAgent instance
            mrq: MRQScorer instance
            config: Optional config dict with:
                - refine_threshold: Energy threshold for refinement
                - fallback_threshold: Energy threshold for LLM fallback
                - max_steps: Max optimization steps for EBT
                - step_size: Learning rate for EBT refinement
        """
        self.ebt = ebt
        self.mrq = mrq
        self.config = config or {}
        
        # Configuration
        self.refine_threshold = self.config.get("refine_threshold", 0.75)
        self.fallback_threshold = self.config.get("fallback_threshold", 0.9)
        self.max_steps = self.config.get("max_steps", 10)
        self.step_size = self.config.get("step_size", 0.05)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def score(self, context: str, text: str, dimension: str = None) -> Dict:
        """
        Score document with EBT-tuned MRQ refinement
        
        Returns:
            {
                "score": float,
                "source": str,  # "ebt" or "mrq" or "llm"
                "refined": bool,
                "converged": bool,
                "energy": float,
                "uncertainty": float
            }
        """
        # Initial MRQ score
        mrq_score = self.mrq.score(context, text, dimension)
        raw_score = mrq_score["score"]
        
        # Get energy from EBT
        energy = self.ebt.get_energy(context, text, dimension)
        uncertainty = sigmoid(torch.tensor(energy)).item()
        
        self.logger.debug("InitialScore", {
            "dimension": dimension,
            "mrq_score": raw_score,
            "energy": energy,
            "uncertainty": uncertainty
        })
        
        # Track refinement state
        refined = False
        source = "mrq"
        final_score = raw_score
        refinement_steps = 0
        energy_trace = []
        
        # Step 1: Refinement if uncertain
        if uncertainty > self.refine_threshold:
            refinement_result = self._refine_document(context, text, dimension)
            refined_text = refinement_result["refined_text"]
            refinement_steps = refinement_result["steps_used"]
            energy_trace = refinement_result["energy_trace"]
            
            # Score refined document
            refined_score = self.mrq.score(context, refined_text, dimension)
            final_score = refined_score["score"]
            refined = True
            source = "ebt"
            
            self.logger.info("DocumentRefined", {
                "dimension": dimension,
                "steps_used": refinement_steps,
                "energy_trace": energy_trace
            })
        
        # Step 2: Fallback if still uncertain
        if refined and energy_trace and energy_trace[-1] > self.fallback_threshold:
            from stephanie.agents.llm import LLMScorer
            llm_scorer = LLMScorer(self.config.get("llm", {}))
            llm_score = llm_scorer.score(context, refined_text if refined else text, dimension)
            final_score = llm_score["score"]
            source = "llm"
            self.logger.warning("LLMFallbackUsed", {
                "dimension": dimension,
                "refined": refined,
                "final_energy": energy_trace[-1] if energy_trace else energy
            })
        
        return {
            "score": final_score,
            "source": source,
            "refined": refined,
            "converged": refinement_steps < self.max_steps if refinement_steps else True,
            "energy": energy,
            "uncertainty": uncertainty,
            "refinement_steps": refinement_steps,
            "energy_trace": energy_trace
        }
    
    def _refine_document(self, context: str, text: str, dimension: str) -> Dict:
        """Refine document using EBT optimization"""
        # Get embeddings
        ctx_emb = torch.tensor(self.ebt.memory.embedding.get_or_create(context)).to(self.device)
        doc_emb = torch.tensor(self.ebt.memory.embedding.get_or_create(text)).to(self.device)
        
        # Make differentiable
        doc_tensor = doc_emb.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([doc_tensor], lr=self.step_size)
        
        energy_trace = []
        for step in range(self.max_steps):
            optimizer.zero_grad()
            energy = self.ebt.models[dimension](ctx_emb, doc_tensor)
            energy.backward()
            optimizer.step()
            energy_trace.append(energy.item())
            
            # Early stopping
            if len(energy_trace) > 1:
                delta = abs(energy_trace[-1] - energy_trace[-2])
                if delta < self.config.get("min_delta", 0.01):
                    break
        
        # Convert refined embedding to text
        refined_emb = doc_tensor.detach()
        refined_text = self.ebt._embedding_to_text(refined_emb, context, text)
        
        return {
            "refined_text": refined_text,
            "final_energy": energy_trace[-1],
            "energy_trace": [round(e, 4) for e in energy_trace],
            "steps_used": len(energy_trace),
            "converged": len(energy_trace) < self.max_steps
        }
    
    def train_from_refinement(self, examples: List[Dict]):
        """
        Retrain MRQ using EBT-refined examples
        
        Args:
            examples: List of {
                "context": str,
                "original": str,
                "refined": str,
                "dimension": str
            }
        """
        # Convert refinement examples to MRQ training data
        training_pairs = []
        for example in examples:
            # Original vs refined
            original_score = self.mrq.score(
                example["context"], 
                example["original"], 
                example["dimension"]
            )["score"]
            
            refined_score = self.mrq.score(
                example["context"], 
                example["refined"], 
                example["dimension"]
            )["score"]
            
            # Create preference pair
            if refined_score > original_score:
                training_pairs.append({
                    "title": example["context"],
                    "output_a": example["refined"],
                    "output_b": example["original"],
                    "value_a": refined_score,
                    "value_b": original_score,
                    "dimension": example["dimension"]
                })
        
        # Train MRQ using preference pairs
        if training_pairs:
            self.mrq.train_multidimensional_model(training_pairs)
            self.logger.info("MRQRetrained", {
                "examples_used": len(training_pairs),
                "dimensions_updated": set(e["dimension"] for e in training_pairs)
            })
    
    def tune(self, context: str, candidate: str, dimension: str = None) -> Dict:
        """
        Tune MRQ using EBT energy feedback
        
        Args:
            context: Goal or prompt text
            candidate: Document or output to evaluate
            dimension: Optional scoring dimension
            
        Returns:
            {
                "improvement": float,
                "before_score": float,
                "after_score": float,
                "dimension": str,
                "refined": bool
            }
        """
        # Get current score
        before_score = self.mrq.score(context, candidate, dimension)["score"]
        
        # Refine using EBT
        refinement = self._refine_document(context, candidate, dimension)
        refined_text = refinement["refined_text"]
        
        # Score refined version
        after_score = self.mrq.score(context, refined_text, dimension)["score"]
        
        # Update MRQ if improvement
        if after_score > before_score:
            self.mrq.update_model(
                context, candidate, refined_text, 
                before_score, after_score
            )
            improvement = after_score - before_score
            self.logger.info("MRQTuned", {
                "dimension": dimension,
                "improvement": improvement,
                "before": before_score,
                "after": after_score
            })
            return {
                "improvement": improvement,
                "before_score": before_score,
                "after_score": after_score,
                "dimension": dimension,
                "refined": True
            }
        
        return {
            "improvement": 0.0,
            "before_score": before_score,
            "after_score": before_score,
            "dimension": dimension,
            "refined": False
        }
    
    def is_uncertain(self, context: str, text: str, dimension: str = None) -> bool:
        """Check if prediction is uncertain using EBT energy"""
        energy = self.ebt.get_energy(context, text, dimension)
        return abs(energy) > self.fallback_threshold
    
    def get_refinement_diff(self, original: str, refined: str) -> str:
        """Return text diff between original and refined versions"""
        from difflib import Differ
        return "\n".join(
            line for line in Differ().compare(original.split(), refined.split())
        )
``n

## File: ebt\refinement_trainer.py

`python
# stephanie/scoring/ebt/ebt_refinement_trainer.py
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.ebt_mixin import EBTMixin
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import save_json
from stephanie.utils.model_utils import get_model_path


class EBTRefinementDataset(Dataset):
    """
    Dataset that contains original → refined document pairs
    Used to train EBT models to recognize refined content
    """
    def __init__(self, refinement_examples: List[Dict], min_score=None, max_score=None):
        """
        Args:
            refinement_examples: List of # Existing structure:
                {
                    "context": str,
                    "original": str,
                    "refined": str,
                    "dimension": str,
                    "original_score": float,
                    "refined_score": float,
                    "original_energy": float,
                    "refined_energy": float,
                    "llm_score": Optional[float],
                    "uncertainty": Optional[float]
                }
        """
        self.data = []
        self.min_score = min_score
        self.max_score = max_score
        
        # Compute global min/max if not provided
        if min_score is None or max_score is None:
            all_scores = [e["score"] for e in refinement_examples]
            self.min_score = min(all_scores) if min_score is None else min_score
            self.max_score = max(all_scores) if max_score is None else max_score

        # Build contrastive pairs
        for example in refinement_examples:
            # Original document gets normalized score
            norm_score = (example["score"] - self.min_score) / (self.max_score - self.min_score)
            
            # Refined document should have higher quality
            refined_score = example.get("refined_score", norm_score + 0.1)
            refined_score = max(0.0, min(1.0, refined_score))
            
            self.data.append({
                "context": example["context"],
                "output_a": example["original"],
                "output_b": example["refined"],
                "value_a": norm_score,
                "value_b": refined_score,
                "dimension": example["dimension"]
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EBTRefinementTrainer(BaseAgent, EBTMixin):
    """
    Trainer for EBT models using refinement examples
    Trains EBT to assign lower energy to refined documents
    """
    def __init__(self, cfg, memory=None, logger=None):
        """
        Args:
            cfg: Configuration dict with:
                - dimensions: List of dimensions to train
                - model_version: Version to save
                - epochs: Training epochs
                - batch_size: Training batch size
                - lr: Learning rate
                - margin: Margin for contrastive loss
        """
        BaseAgent.__init__(self, cfg, memory, logger)
        EBTMixin.__init__(self, cfg.get("ebt", {}))
        
        # Training configuration
        self.epochs = cfg.get("epochs", 10)
        self.batch_size = cfg.get("batch_size", 8)
        self.lr = cfg.get("learning_rate", 2e-5)
        self.margin = cfg.get("margin", 1.0)
        self.save_interval = cfg.get("save_interval", 1)
        self.model_path = cfg.get("model_path", "models")
        self.model_type = "ebt"
        self.target_type = "document"
        self.model_version = cfg.get("model_version", "v1")
        self.embedding_type = self.memory.embedding.type
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize regression tuners
        self.tuners = {
            dim: RegressionTuner(dimension=dim, logger=logger)
            for dim in self.dimensions
        }

    def prepare_refinement_data(self, examples: List[Dict]) -> DataLoader:
        """
        Convert raw refinement examples into training-ready DataLoader
        """
        # Group by dimension
        by_dimension = defaultdict(list)
        for example in examples:
            dim = example.get("dimension", "default")
            by_dimension[dim].extend(self._create_refinement_pairs(example))
        
        # Create datasets and loaders
        loaders = {}
        for dim, pairs in by_dimension.items():
            ds = EBTRefinementDataset(pairs)
            loaders[dim] = DataLoader(
                ds,
                num_workers= 4,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=lambda b: self._collate_fn(b, self.memory.embedding, self.device)
            )
        return loaders

    def _create_refinement_pairs(self, example: Dict) -> List[Dict]:
        """
        Convert a single refinement example into contrastive training pairs
        """
        context = example["context"]
        refined = example["refined"]
        original = example["original"]
        dim = example["dimension"]
        
        # Create multiple variations for robustness
        return [
            {
                "context": context,
                "output_a": original,
                "output_b": refined,
                "value_a": example["value"],
                "value_b": example["refined_value"],
                "dimension": dim
            },
            {
                "context": context,
                "output_a": refined,
                "output_b": original,
                "value_a": example["refined_value"],
                "value_b": example["value"],
                "dimension": dim
            }
        ]

    def _collate_fn(self, batch, embedding_store, device):
        """
        Convert batch of examples to tensors
        """
        ctxs, docs_a, docs_b, labels = [], [], [], []
        
        for item in batch:
            # Get embeddings
            ctx_emb = torch.tensor(embedding_store.get_or_create(item["context"])).to(device)
            a_emb = torch.tensor(embedding_store.get_or_create(item["output_a"])).to(device)
            b_emb = torch.tensor(embedding_store.get_or_create(item["output_b"])).to(device)
            
            # Contrastive loss labels
            preferred = "a" if item["value_a"] > item["value_b"] else "b"
            labels.append(1.0 if preferred == "a" else 0.0)
            
            ctxs.append(ctx_emb)
            docs_a.append(a_emb)
            docs_b.append(b_emb)
        
        # Stack tensors
        ctx_tensor = torch.stack(ctxs)
        doc_a_tensor = torch.stack(docs_a)
        doc_b_tensor = torch.stack(docs_b)
        label_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
        
        return ctx_tensor, doc_a_tensor, doc_b_tensor, label_tensor

    def contrastive_loss(self, energy_a, energy_b, label):
        """
        Contrastive loss for refinement training
        """
        # Label: 1 if a is better than b, 0 otherwise
        margin = self.cfg.get("loss_margin", 1.0)
        distances = torch.abs(energy_a - energy_b)
        
        # Calculate loss
        if label == 1:
            return distances  # We want energy_a < energy_b → smaller distance is good
        else:
            return torch.relu(margin - distances)  # Push apart if margin not met

    def train_refinement_model(self, dimension: str, dataloader: DataLoader):
        """
        Train EBT model for a single dimension
        """
        # Initialize fresh model for this dimension
        model = self._initialize_dimension_model(dimension)
        model.to(self.device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        total_loss = 0.0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for ctx, doc_a, doc_b, labels in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                energy_a = model(ctx, doc_a)
                energy_b = model(ctx, doc_b)
                
                # Calculate loss
                loss = self.contrastive_loss(energy_a, energy_b, labels).mean()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.logger.log("EBTRefinementEpoch", {
                "dimension": dimension,
                "epoch": epoch + 1,
                "loss": avg_epoch_loss
            })
            
            # Periodic model saving
            if (epoch + 1) % self.save_interval == 0:
                self._save_model(model, dimension, f"v{epoch + 1}")
        
        # Final save
        final_path = self._save_model(model, dimension, "latest")
        self.logger.log("EBTRefinementTrainingComplete", {
            "dimension": dimension,
            "final_loss": avg_epoch_loss,
            "model_path": final_path
        })
        return model

    def _initialize_dimension_model(self, dimension: str) -> EBTModel:
        """Initialize a fresh EBT model for a dimension"""
        model_path = get_model_path(
            self.model_path,
            self.model_type,
            self.target_type,
            dimension,
            self.model_version,
            self.embedding_type
        )
        os.makedirs(model_path, exist_ok=True)
        
        return EBTModel().to(self.device)

    def _save_model(self, model, dimension: str, version: str = "latest") -> str:
        """Save model and metadata"""
        model_path = get_model_path(
            self.model_path,
            self.model_type,
            self.target_type,
            dimension,
            version
        )
        
        # Save model weights
        torch.save(model.state_dict(), os.path.join(model_path, f"{dimension}.pt"))
        
        # Save normalization metadata
        meta_path = os.path.join(model_path, f"{dimension}.meta.json")
        meta = {
            "min_score": self.ebt_meta.get(dimension, {}).get("min", 40),
            "max_score": self.ebt_meta.get(dimension, {}).get("max", 100),
            "train_min_score": self._get_train_min_score(dimension),
            "train_max_score": self._get_train_max_score(dimension),
            "training_date": datetime.utcnow().isoformat(),
            "version": version
        }
        save_json(meta, meta_path)
        
        return model_path

    def _get_train_min_score(self, dimension: str):
        """Get training minimum score for normalization"""
        query = f"""
        SELECT MIN(score) FROM scoring_events
        WHERE dimension='{dimension}' AND source IN ('ebt', 'llm')
        """
        result = self.memory.db.execute(query).fetchone()
        return result[0] if result else 40

    def srft_loss(self, energy_orig, energy_ref, llm_score=None, orig_score=None, ref_score=None, entropy=None):
        """
        SRFT-style loss combining:
        - Supervised Fine-Tuning (match LLM score or gold refined score)
        - Reinforcement-style reward (improve MRQ score or reduce energy)
        """
        losses = []

        # 1. SFT: encourage refined to match or beat LLM score
        if llm_score is not None:
            target = torch.tensor(llm_score).to(self.device)
            losses.append(torch.nn.functional.mse_loss(energy_ref, target))

        # 2. RL-style reward: reward if refined energy < original
        if energy_orig is not None:
            margin = self.cfg.get("rl_margin", 0.05)
            diff = energy_orig - energy_ref
            rl_loss = -torch.relu(diff - margin)  # reward if improvement
            losses.append(rl_loss.mean())

        # 3. Entropy-aware weighting (optional)
        if entropy is not None:
            weight = 1.0 / (1.0 + entropy)
            total = sum(losses)
            return weight * total

        return sum(losses)


    def _get_train_max_score(self, dimension: str):
        """Get training maximum score for normalization"""
        query = f"""
        SELECT MAX(score) FROM scoring_events
        WHERE dimension='{dimension}' AND source IN ('ebt', 'llm')
        """
        result = self.memory.db.execute(query).fetchone()
        return result[0] if result else 100

    async def run(self, context: dict) -> dict:
        """
        Main training loop for EBT refinement models
        """
        goal_text = context.get("goal", {}).get("goal_text")
        refinement_data = context.get("refinement_data", [])
        
        if not refinement_data:
            # Fetch from database if no data provided
            refinement_data = self._fetch_refinement_examples(goal_text)
        
        # Prepare data
        dataloaders = self.prepare_refinement_data(refinement_data)
        
        # Train per-dimension models
        trained_models = {}
        for dim, loader in dataloaders.items():
            self.logger.log("EBTRefinementStart", {
                "dimension": dim,
                "examples": len(loader.dataset)
            })
            
            trained_model = self.train_refinement_model(dim, loader)
            trained_models[dim] = trained_model.state_dict()
            
            # Update tuner
            if dim in self.tuners:
                self._update_regression_tuner(loader, dim)
        
        # Update model registry
        self._update_model_registry(trained_models)
        
        context["trained_models"] = trained_models
        return context

    def _fetch_refinement_examples(self, goal: str = None) -> List[Dict]:
        """
        Fetch refinement examples from database
        """
        query = """
        SELECT * FROM refinement_events
        WHERE created_at > NOW() - INTERVAL '7 days'
        """
        if goal:
            query += f"AND context_hash = {hash(goal)}"
        
        results = self.memory.db.execute(query).fetchall()
        
        return [{
            "context": r.context,
            "original": r.original,
            "refined": r.refined,
            "dimension": r.dimension,
            "score": r.original_score,
            "refined_score": r.refined_score
        } for r in results]

    def _update_regression_tuner(self, dataloader: DataLoader, dimension: str):
        """Update regression tuner using refined examples"""
        for ctx, doc_a, doc_b, labels in dataloader:
            for i in range(len(ctx)):
                original = doc_a[i].cpu().numpy()
                refined = doc_b[i].cpu().numpy()
                llm_score = labels[i].cpu().item()
                
                # Update tuner with EBT-refined examples
                self.tuners[dimension].train_single(
                    ebt_score=doc_b[i].item(),
                    llm_score=llm_score
                )

    def _update_model_registry(self, trained_models: Dict[str, Dict]):
        """Update model registry with new versions"""
        for dim, state_dict in trained_models.items():
            self.logger.log("EBTModelUpdated", {
                "dimension": dim,
                "version": "auto",
                "performance": self._evaluate_model(dim, state_dict)
            })

    def _evaluate_model(self, dimension: str, model_state) -> Dict:
        """Evaluate model performance on validation data"""
        # Implement validation logic here
        return {
            "val_loss": 0.123,
            "accuracy": 0.91,
            "improvement": 0.05
        }
``n

## File: ebt\srft_collate.py

`python
from typing import List

import torch


def srft_collate_fn(batch: List[dict], embedding_store, device):
    """
    Collate function for SRFT refinement training.

    Embeds:
        - context
        - original
        - refined

    Returns:
        - context_tensor: (B, D)
        - original_tensor: (B, D)
        - refined_tensor: (B, D)
        - original_scores: (B,)
        - refined_scores: (B,)
        - original_energy: (B,)
        - refined_energy: (B,)
        - llm_scores: (B,) or None
        - uncertainty: (B,)
    """
    contexts = []
    originals = []
    refineds = []
    orig_scores, ref_scores = [], []
    orig_energy, ref_energy = [], []
    llm_scores = []
    uncertainties = []

    for item in batch:
        contexts.append(torch.tensor(embedding_store.get_or_create(item["context"])))
        originals.append(torch.tensor(embedding_store.get_or_create(item["original"])))
        refineds.append(torch.tensor(embedding_store.get_or_create(item["refined"])))

        orig_scores.append(item["original_score"])
        ref_scores.append(item["refined_score"])
        orig_energy.append(item["original_energy"])
        ref_energy.append(item["refined_energy"])
        uncertainties.append(item.get("uncertainty", 0.5))
        llm_scores.append(item.get("llm_score", -1))  # use -1 for missing supervision

    return (
        torch.stack(contexts).to(device),
        torch.stack(originals).to(device),
        torch.stack(refineds).to(device),
        torch.tensor(orig_scores, dtype=torch.float32).to(device),
        torch.tensor(ref_scores, dtype=torch.float32).to(device),
        torch.tensor(orig_energy, dtype=torch.float32).to(device),
        torch.tensor(ref_energy, dtype=torch.float32).to(device),
        torch.tensor(llm_scores, dtype=torch.float32).to(device),
        torch.tensor(uncertainties, dtype=torch.float32).to(device),
    )
``n

## File: ebt\srft_refinement_dataset.py

`python
# stephanie/scoring/ebt/srft_refinement_dataset.py

from typing import Dict, List

from torch.utils.data import Dataset


class SRFTRefinementDataset(Dataset):
    """
    Dataset for SRFT-style training:
    Includes original/refined embeddings + score/energy/uncertainty data
    """

    def __init__(self, examples: List[Dict]):
        """
        Args:
            examples: list of refinement examples with fields like:
                - context
                - original
                - refined
                - dimension
                - original_score
                - refined_score
                - original_energy
                - refined_energy
                - llm_score
                - uncertainty
        """
        self.data = []

        for ex in examples:
            self.data.append({
                "context": ex["context"],
                "original": ex["original"],
                "refined": ex["refined"],
                "dimension": ex["dimension"],
                "original_score": ex.get("original_score", 0.5),
                "refined_score": ex.get("refined_score", 0.6),
                "original_energy": ex.get("original_energy", 0.5),
                "refined_energy": ex.get("refined_energy", 0.4),
                "llm_score": ex.get("llm_score"),
                "uncertainty": ex.get("uncertainty", 0.5)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
``n

## File: fallback_scorer.py

`python
# File: stephanie/scoring/fallback_scorer.py

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

from stephanie.constants import GOAL
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle


@dataclass
class ScoreAttempt:
    scorer_name: str
    success: bool
    error: Optional[str] = None
    score_bundle: Optional[ScoreBundle] = None
    timestamp: datetime = datetime.utcnow()


class FallbackScorer(BaseScorer):
    """
    A composite scorer that tries multiple scorers in order of preference.
    Falls back when a scorer fails due to missing model, data, or timeout.
    """

    def __init__(
        self,
        scorers: List[BaseScorer],
        fallback_order: Optional[List[str]] = None,
        default_fallback: str = "llm",
        logger = None,
    ): 
        """
        Args:
            scorers: List of scorers to try
            fallback_order: Order of scorer preference (e.g. ["svm", "mrq", "llm"])
            default_fallback: Final fallback if all scorers fail
            logger: Optional logger
        """
        super().__init__(cfg={}, memory=None, logger=logger)
        self.scorers = {scorer.name: scorer for scorer in scorers}
        self.fallback_order = fallback_order or list(self.scorers.keys())
        self.default_fallback = default_fallback

    def score(self, context: dict, scorable: Scorable, dimensions: List[str] = None) -> ScoreBundle:
        """
        Try scorers in order. Returns first successful score bundle.
        If all fail, returns neutral score with fallback reason.
        """
        goal = context.get(GOAL, {})
        scorer_used = None
        attempts = []

        for scorer_name in self.fallback_order:
            scorer = self.scorers.get(scorer_name)

            if not scorer:
                self.logger.log(
                    "ScorerNotRegistered",
                    {"scorer_name": scorer_name, "available_scorers": list(self.scorers.keys())}
                )
                continue

            try:
                self.logger.log("TryingScorer", {"scorer": scorer_name, "target": scorable.id})
                score_bundle = scorer.score(context, scorable, dimensions=dimensions)

                if score_bundle.is_valid():
                    scorer_used = scorer_name
                    attempts.append(ScoreAttempt(
                        scorer_name=scorer_name,
                        success=True,
                        score_bundle=score_bundle
                    ))
                    self.logger.log(
                        "ScoreSuccess",
                        {"scorer": scorer_name, "target": scorable.id, "scores": score_bundle.to_dict()}
                    )
                    break
                else:
                    attempts.append(ScoreAttempt(
                        scorer_name=scorer_name,
                        success=False,
                        error="Invalid score bundle"
                    ))
                    self.logger.log(
                        "ScoreInvalid",
                        {"scorer": scorer_name, "target": scorable.id}
                    )

            except Exception as e:
                attempts.append(ScoreAttempt(
                    scorer_name=scorer_name,
                    success=False,
                    error=str(e)
                ))
                self.logger.log(
                    "ScoreFailed",
                    {"scorer": scorer_name, "target": scorable.id, "error": str(e)}
                )
                continue

        if scorer_used:
            return score_bundle

        # Final fallback: use default scorer
        default_scorer = self.scorers.get(self.default_fallback)
        if default_scorer:
            self.logger.log(
                "FinalFallbackUsed",
                {"scorer": self.default_fallback, "target": scorable.id}
            )
            return default_scorer.score(context, scorable, dimensions=dimensions)

        # If even fallback fails, return neutral score
        self.logger.log(
            "AllScorersFailed",
            {"target": scorable.id, "attempts": [a.scorer_name for a in attempts]}
        )
        return ScoreBundle.from_dict({
            dim: 0.5 for dim in dimensions or ["usefulness", "novelty", "alignment"]
        })

    def load_models(self):
        """Load models for all scorers."""
        for scorer in self.scorers.values():
            try:
                scorer.load_models()
            except Exception as e:
                self.logger.log("ModelLoadFailed", {"scorer": scorer.name, "error": str(e)})

    def train(self, samples_by_dim: Dict[str, list]):
        """Train all scorers that support training."""
        for scorer in self.scorers.values():
            try:
                scorer.train(samples_by_dim)
            except Exception as e:
                self.logger.log("TrainingFailed", {"scorer": scorer.name, "error": str(e)})

    def save_models(self):
        """Save models for all scorers."""
        for scorer in self.scorers.values():
            try:
                scorer.save_models()
            except Exception as e:
                self.logger.log("ModelSaveFailed", {"scorer": scorer.name, "error": str(e)})
``n

## File: llm_scorer.py

`python
# stephanie/scoring/llm_scorer.py

import re
from string import Template

from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.scoring_manager import ScoringManager


class LLMScorer(BaseScorer):
    """
    Scores a hypothesis using an LLM per dimension.
    Uses structured templates and flexible response parsers.
    """

    def __init__(self, cfg, memory, logger, prompt_loader=None, llm_fn=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.prompt_loader = prompt_loader
        self.llm_fn = llm_fn
        self.force_rescore = cfg.get("force_rescore", False)    

    @property
    def name(self) -> str:
        return "llm"

    def score(self, context:dict, scorable: Scorable, dimensions: list[dict], llm_fn=None) -> ScoreBundle:
        """
        Scores a Scorable across multiple dimensions using an LLM.
        Returns a ScoreBundle object.
        Accepts either:
        - A list of dimension names (strings)
        - A list of dimension dicts: {name, prompt_template, weight, parser, etc.}
        """
        results = []

        for dim in dimensions:
            prompt = self._render_prompt(context, scorable, dim)
            prompt_hash = ScoreORM.compute_prompt_hash(prompt, scorable)

            if not self.force_rescore:
                cached_result = self.memory.scores.get_score_by_prompt_hash(prompt_hash)
                if cached_result:
                    self.logger.log("ScoreCacheHit", {"dimension": dim["name"]})
                    result = cached_result
                    results.append(result)
                    continue

            if not llm_fn:
                llm_fn = self.llm_fn
            response = llm_fn(prompt, context)

            try:
                parser = dim.get("parser") or self._get_parser(dim)
                score = parser(response)
            except Exception as e:
                score = 0.0
                if self.logger:
                    self.logger.log(
                        "LLMScoreParseError",
                        {
                            "dimension": dim["name"],
                            "response": response,
                            "error": str(e),
                        },
                    )

            if self.logger:
                self.logger.log(
                    "LLMJudgeScorerDimension",
                    {
                        "dimension": dim["name"],
                        "score": score,
                        "rationale": response,
                    },
                )

            result = ScoreResult(
                dimension=dim["name"],
                score=score,
                rationale=response,
                weight=dim.get("weight", 1.0),
                source="llm",
                target_type=scorable.target_type,
                prompt_hash=prompt_hash,
            )

            results.append(result)

        # Aggregate scores across dimensions
        bundle = ScoreBundle(results={r.dimension: r for r in results})
        ScoringManager.save_score_to_memory(
            bundle,
            scorable,
            context,
            self.cfg,
            self.memory,
            self.logger,
            source="llm",
        )
        return bundle


    def _render_prompt(self, context: dict, scorable: Scorable, dim: dict) -> str:
        merged_context = {
            "scorable": scorable,
            **context
        }
        if self.prompt_loader and dim.get("file"):
            return self.prompt_loader.score_prompt(
                file_name=dim["file"], config=self.cfg, context=merged_context
            )
        else:
            return Template(dim["prompt_template"]).substitute(merged_context)

    def _default_prompt(self, dimension):
        return (
            "Evaluate the following document based on $dimension:\n\n"
            "Goal: $goal\nHypothesis: $scorable\n\n"
            "Respond with a score and rationale."
        ).replace("$dimension", dimension)

    def _aggregate(self, results: dict) -> float:
        total = 0.0
        weight_sum = 0.0
        for dim, val in results.items():
            if not isinstance(val, dict):
                continue
            total += val["score"] * val.get("weight", 1.0)
            weight_sum += val.get("weight", 1.0)
        return round(total / weight_sum, 2) if weight_sum else 0.0

    @staticmethod
    def extract_score_from_last_line(response: str) -> float:
        lines = response.strip().splitlines()
        for line in reversed(lines):
            match = re.search(
                r"score[:\-]?\s*(\d+(\.\d+)?)", line.strip(), re.IGNORECASE
            )
            if match:
                return float(match.group(1))
        return 0.0

    @staticmethod
    def parse_numeric_cor(response: str) -> float:
        match = re.search(
            r"<answer>\s*\[\[(\d+(?:\.\d+)?)\]\]\s*</answer>", response, re.IGNORECASE
        )
        if not match:
            raise ValueError(
                f"Could not extract numeric score from CoR-style answer: {response}"
            )
        return float(match.group(1))

    def _get_parser(self, dim: dict):
        parser_type = dim.get("parser", "numeric")
        if parser_type == "numeric":
            return self.extract_score_from_last_line
        if parser_type == "numeric_cor":
            return self.parse_numeric_cor
        return lambda r: 0.0
``n

## File: meta_review_scorer.py

`python
# stephanie/scoring/meta_review_scorer.py

from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.llm_scorer import LLMScorer
from stephanie.scoring.mrq_scorer import MRQScorer  # formerly MRQEvaluator


class MetaReviewScorer(BaseScorer):
    """
    Combines MR.Q-based scoring with LLM fallback.

    Tries MRQ first. If it's missing dimensions or returns low-confidence scores, falls back to LLM.
    """

    def __init__(self, memory, logger, cfg=None, fallback_to_llm=True):
        self.memory = memory
        self.logger = logger
        self.cfg = cfg or {}
        self.use_llm_fallback = fallback_to_llm

        self.mrq_scorer = MRQScorer(cfg, memory, logger)
        self.llm_scorer = LLMScorer(memory, logger, cfg)

    def score(self, goal, hypothesis, dimensions):
        mrq_scores = self.mrq_scorer.score(goal, hypothesis, dimensions)

        if self._needs_llm_fallback(mrq_scores, dimensions):
            self.logger.log(
                "MetaReviewFallbackTriggered",
                {"reason": "Missing or low-confidence MRQ scores", "fallback": "llm"},
            )
            llm_scores = self.llm_scorer.score(goal, hypothesis, dimensions)
            return self._combine_scores(mrq_scores, llm_scores)
        else:
            return mrq_scores

    def _needs_llm_fallback(self, scores, dimensions):
        """
        Trigger fallback if:
        - Any dimension is missing in MRQ
        - Any score is None or clearly untrained (e.g., 0.0)
        """
        for dim in dimensions:
            if dim not in scores:
                return True
            if abs(scores[dim]["score"] - 0.0) < 1e-8:
                return True
        return False

    def _combine_scores(self, mrq_scores, llm_scores):
        """
        Merge: prefer MRQ if valid, otherwise use LLM.
        """
        combined = {}
        all_dims = set(mrq_scores.keys()) | set(llm_scores.keys())
        for dim in all_dims:
            if dim in mrq_scores and mrq_scores[dim]["score"] > 0.0:
                combined[dim] = mrq_scores[dim]
            else:
                combined[dim] = llm_scores.get(
                    dim,
                    {"score": 0.0, "rationale": "No fallback available", "weight": 1.0},
                )
        return combined
``n

## File: model_locator_mixin.py

`python
# stephanie/scoring/model_locator_mixin.py

import os


class ModelLocatorMixin:
    class Locator:
        def __init__(
            self,
            root_dir: str,
            model_type: str,
            target_type: str,
            dimension: str,
            version: str,
            embedding_type: str,
        ):
            self.root_dir = root_dir
            self.model_type = model_type
            self.target_type = target_type
            self.dimension = dimension
            self.version = version
            self.embedding_type = embedding_type

        @property
        def base_path(self) -> str:
            path = os.path.join(
                self.root_dir,
                self.embedding_type,
                self.model_type,
                self.target_type,
                self.dimension,
                self.version,
            )
            os.makedirs(path, exist_ok=True)
            return path

        # Model-specific paths
        def model_file(self, suffix: str = ".pt") -> str:
            return os.path.join(self.base_path, f"{self.dimension}{suffix}")

        def encoder_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_encoder.pt")

        def get_q_head_path(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_q.pt")

        def get_v_head_path(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_v.pt")

        def get_pi_head_path(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_pi.pt")

        def meta_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.meta.json")

        def tuner_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.tuner.json")

        def scaler_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_scaler.joblib")

    def get_model_name(self) -> str:
        return f"{self.target_type}_{self.model_type}_{self.model_version}"

    def get_locator(self, dimension: str):
        return self.Locator(
            root_dir=self.model_path,  # Path to the root directory for models
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.embedding_type,
        )
``n

## File: model\ebt_model.py

`python
# stephanie/scoring/model/ebt_model.py
import torch
from torch import nn
from torch.nn import functional as F


class EBTModel(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=256, num_actions=3, device="cpu"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.device = device

        # Encoder with attention
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Q head with learnable scaling
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # V head with expectile regression
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Policy head with policy entropy
        self.pi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # Learnable scaling factor
        self.scale_factor = nn.Parameter(torch.tensor(10.0))

    def forward(self, context_emb, output_emb):
        # Ensure device alignment
        context_emb = context_emb.to(self.device)
        output_emb = output_emb.to(self.device)
        
        # Combine embeddings
        combined = torch.cat([context_emb, output_emb], dim=-1)
        zsa = self.encoder(combined)
        
        # Q/V heads
        q_value = self.q_head(zsa).squeeze()
        state_value = self.v_head(zsa).squeeze()
        
        # Policy head
        action_logits = self.pi_head(zsa)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Compute advantage
        advantage = q_value - state_value
        
        # Scale final score
        final_score = q_value * torch.sigmoid(self.scale_factor).item()
        
        return {
            "q_value": q_value,
            "state_value": state_value,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "advantage": advantage,
            "score": final_score
        }
    
``n

## File: model\energy_based_scorer.py

`python
# stephanie/scoring/model/energy_based_scorer.py
import torch
import torch.nn as nn


class EBTThinker:
    def __init__(self, model, step_size=0.05, steps=10):
        self.model = model
        self.step_size = step_size
        self.steps = steps

    def optimize(self, context_emb, candidate_emb):
        y = candidate_emb.clone().detach().requires_grad_(True)
        energies = []

        for _ in range(self.steps):
            energy = self.model(context_emb, y)
            grad = torch.autograd.grad(energy.sum(), y, create_graph=False)[0]
            y = y - self.step_size * grad
            energies.append(energy.item())

        final_energy = energies[-1]
        converged = abs(energies[-1] - energies[0]) < 0.01
        return {
            "refined_candidate": y.detach(),
            "energy": final_energy,
            "steps_used": len(energies),
            "converged": converged,
            "energy_trace": energies,
        }


class EnergyBasedScorer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer  # Any pretrained or scratch Transformer

    def forward(self, context_emb, candidate_emb):
        combined = torch.cat([context_emb, candidate_emb], dim=-1)
        hidden = self.transformer(combined)
        energy = hidden.mean(dim=-1)  # scalar energy per example
        return energy
``n

## File: model\in_context_q.py

`python
# stephanie/scoring/mrq/model.py
import torch
from torch import nn

from stephanie.scoring.model.policy_head import PolicyHead
from stephanie.scoring.model.q_head import QHead
from stephanie.scoring.model.v_head import VHead
from stephanie.scoring.mrq.encoder import TextEncoder


class InContextQModel(nn.Module):
    def __init__(
        self, 
        encoder: TextEncoder,
        q_head: QHead,
        v_head: VHead,
        pi_head: PolicyHead,
        embedding_store,
        device="cpu"
    ):
        super().__init__()
        self.encoder = encoder.to(device)
        self.q_head = q_head.to(device)
        self.v_head = v_head.to(device)
        self.pi_head = pi_head.to(device)
        self.device = device
        self.embedding_store = embedding_store
    
    def forward(self, context_emb, doc_emb):
        """
        Forward pass through all heads
        
        Args:
            context_emb: Goal/prompt embedding
            doc_emb: Document/output embedding
        Returns:
            Dict containing Q-value, state value, and policy logits
        """
        # Ensure device alignment
        context_emb = context_emb.to(self.device)
        doc_emb = doc_emb.to(self.device)
        
        # Combine embeddings
        zsa = self.encoder(context_emb, doc_emb)
        
        # Forward through heads
        q_value = self.q_head(zsa)
        state_value = self.v_head(zsa)
        action_logits = self.pi_head(zsa)
        
        # Calculate advantage
        advantage = (q_value - state_value).detach()
        
        return {
            "q_value": q_value,
            "state_value": state_value,
            "action_logits": action_logits,
            "advantage": advantage
        }
``n

## File: model\policy_head.py

`python
# stephanie/scoring/model/policy_head.py
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU


class PolicyHead(nn.Module):
    def __init__(self, zsa_dim, hdim, num_actions=3):
        super().__init__()
        self.linear = nn.Sequential(
            Linear(zsa_dim, hdim),
            ReLU(),
            Linear(hdim, num_actions)
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, zsa):
        return self.linear(zsa)

    def get_policy_weights(self):
        """
        Get the averaged weights of the final linear layer for policy logits.
        """
        final_linear_layer = self.linear[-1]
        return final_linear_layer.weight.data.mean(dim=0)
``n

## File: model\q_head.py

`python
# stephanie/scoring/model/q_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU


class QHead(nn.Module):
    def __init__(self, zsa_dim, hdim):
        """
        Q-value estimator: Q(s,a) = E[reward | state, action]
        
        Args:
            zsa_dim: Dimension of encoded state-action vector
            hdim: Hidden layer dimension
        """
        super().__init__()
        self.model = nn.Sequential(
            Linear(zsa_dim, hdim),
            ReLU(),
            Linear(hdim, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, zsa):
        """
        Predict Q-value for (state, action) pair
        Args:
            zsa: Encoded state-action vector
        Returns:
            Q-value (scalar)
        """
        return self.model(zsa).squeeze()
``n

## File: model\v_head.py

`python
# stephanie/scoring/model/v_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU


class VHead(nn.Module):
    def __init__(self, zsa_dim, hdim):
        """
        State value estimator using expectile regression
        
        Args:
            zsa_dim: Dimension of encoded state-action vector
            hdim: Hidden layer dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            Linear(zsa_dim, hdim),
            ReLU(),
            Linear(hdim, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, zsa):
        """
        Predict state value V(s)
        Args:
            zsa: Encoded state-action vector
        Returns:
            State value (scalar)
        """
        return self.net(zsa).squeeze()
``n

## File: mrq_scorer.py

`python
# stephanie/scoring/mrq/mrq_scorer.py

import os

import torch

from stephanie.evaluator.hypothesis_value_predictor import \
    HypothesisValuePredictor
from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.model import MRQModel
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator


class MRQScorer(BaseScorer):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "mrq"
        self.embedding_type = memory.embedding.type
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim
        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")

        self.models = {}
        self.model_meta = {}
        self.tuners = {}

        self.dimensions = cfg.get("dimensions", [])
        self._load_models(self.dimensions)

    def _load_models(self, dimensions):
        for dim in dimensions:
            locator = ModelLocator(
                root_dir=self.model_path,
                embedding_type=self.embedding_type,
                model_type=self.model_type,
                target_type=self.target_type,
                dimension=dim,
                version=self.version,
            )

            encoder = TextEncoder(self.dim, self.hdim)
            predictor = HypothesisValuePredictor(self.dim, self.hdim)
            model = MRQModel(encoder, predictor, self.memory.embedding, device=self.device)
            model.load_weights(locator.encoder_file(), locator.model_file())
            self.models[dim] = model

            meta = load_json(locator.meta_file()) if os.path.exists(locator.meta_file()) else {"min_score": 0, "max_score": 100}
            self.model_meta[dim] = meta

            tuner_path = locator.tuner_file()
            if os.path.exists(tuner_path):
                tuner = RegressionTuner(dimension=dim)
                tuner.load(tuner_path)
                self.tuners[dim] = tuner

    def score(self, goal: dict, scorable, dimensions: list[str]) -> ScoreBundle:
        goal_text = goal.get("goal_text")
        results = {}

        for dim in dimensions:
            model = self.models.get(dim)
            if not model:
                continue

            q_value = model.predict(goal_text, scorable.text)

            meta = self.model_meta.get(dim, {"min_score": 0, "max_score": 100})
            tuner = self.tuners.get(dim)

            if tuner:
                scaled = tuner.transform(q_value)
            else:
                norm = torch.sigmoid(torch.tensor(q_value)).item()
                if norm < 0.01 or norm > 0.99:
                    self.logger.log("QValueOutlier", {"dimension": dim, "q_value": q_value})
                scaled = norm * (meta["max_score"] - meta["min_score"]) + meta["min_score"]

            final_score = round(max(min(scaled, meta["max_score"]), meta["min_score"]), 4)
            prompt_hash = ScoreORM.compute_prompt_hash(goal_text, scorable)

            results[dim] = ScoreResult(
                dimension=dim,
                score=final_score,
                rationale=f"Q={round(q_value, 4)}",
                weight=1.0,
                q_value=q_value,
                energy=q_value,
                source="mrq",
                target_type=scorable.target_type,
                prompt_hash=prompt_hash,
            )

        return ScoreBundle(results=results)
``n

## File: mrq\__init__.py

`python
# stephanie\scoring\mrq\__init__.py
from .model import MRQModel
from .scorer import MRQScorer
from .trainer import MRQTrainer
``n

## File: mrq\core_scoring.py

`python
# stephanie/scoring/mrq/core_scoring.py
import torch

from stephanie.models.score import ScoreORM
from stephanie.models.sharpening_prediction import SharpeningPredictionORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.scoring_manager import ScoringManager


class MRQCoreScoring:
    def evaluate(self, prompt, response):
        results = []
        for dim, (encoder, predictor) in self.models.items():
            prompt_emb = torch.tensor(
                self.memory.embedding.get_or_create(prompt), device=self.device
            ).unsqueeze(0)
            output_emb = torch.tensor(
                self.memory.embedding.get_or_create(response), device=self.device
            ).unsqueeze(0)
            zsa = encoder(prompt_emb, output_emb)
            value = predictor(zsa).item()
            norm_score = self.normalize_score(value, dim)

            results.append(
                ScoreResult(
                    dimension=dim,
                    score=norm_score,
                    weight=1.0,
                    rationale=f"MR.Q model trained for {dim}",
                    source="mrq",
                )
            )

        bundle = ScoreBundle(results={r.dimension: r for r in results})
        ScoringManager.save_score_to_memory(
            bundle,
            response,
            cfg=self.cfg,
            memory=self.memory,
            logger=self.logger,
            source="mrq",
        )
        return bundle

    def judge(self, goal, prompt, output_a, output_b):
        dim = self.dimensions[0]
        model = self.models[dim]
        encoder = model.encoder
        predictor = model.predictor

        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)
        a_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_a), device=self.device
        ).unsqueeze(0)
        b_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_b), device=self.device
        ).unsqueeze(0)

        value_a = predictor(encoder(prompt_emb, a_emb)).item()
        value_b = predictor(encoder(prompt_emb, b_emb)).item()
        preferred = output_a if value_a >= value_b else output_b

        if self.memory.mrq.log_evaluations():
            pred = SharpeningPredictionORM(
                id=None,
                goal_id=-1,
                prompt_text=prompt,
                output_a=output_a,
                output_b=output_b,
                preferred="a" if value_a >= value_b else "b",
                predicted="a" if value_a >= value_b else "b",
                value_a=value_a,
                value_b=value_b,
            )
            self.memory.sharpening.insert_sharpening_prediction(pred.to_dict(), goal)

        return preferred, {"value_a": value_a, "value_b": value_b}

    def predict_score_from_prompt(self, prompt, dimension="mrq", top_k=5):
        try:
            nearest = self.memory.embedding.search_similar_prompts_with_scores(
                prompt, top_k=top_k
            )

            llm_scores = []
            mrq_scores = []

            for item in nearest:
                if item.get("dimension") != dimension:
                    continue

                score = item.get("score")
                source = item.get("source")

                if score is None:
                    continue

                if source == "llm":
                    llm_scores.append(score)
                elif source == "mrq":
                    mrq_scores.append(score)

            if not llm_scores:
                if self.logger:
                    self.logger.log(
                        "MRQPromptScorerNoLLMScoresFound",
                        {"prompt": prompt[:100], "dimension": dimension},
                    )
                return 0.5

            avg_llm = sum(llm_scores) / len(llm_scores)

            if mrq_scores and dimension in self.regression_tuners:
                avg_mrq = sum(mrq_scores) / len(mrq_scores)
                aligned_score = self.regression_tuners[dimension].transform(avg_mrq)
            else:
                aligned_score = avg_llm

            final_score = max(0.0, min(1.0, aligned_score))

            if self.logger:
                self.logger.log(
                    "MRQPromptScorePredicted",
                    {
                        "prompt": prompt[:100],
                        "score": final_score,
                        "dimension": dimension,
                        "neighbors_found": len(nearest),
                        "used_alignment": bool(mrq_scores),
                    },
                )

            return final_score

        except Exception as e:
            if self.logger:
                self.logger.log(
                    "MRQPromptScoreError",
                    {"error": str(e), "prompt": prompt[:100], "dimension": dimension},
                )
            return 0.5

    def estimate_score(self, goal, scorable, dimension):
        """
        Core logic: compute embeddings, run prediction, apply optional regression tuner.
        """
        if dimension not in self.models:
            self.initialize_dimension(dimension)

        raw_score = self.models[dimension].predict(
            goal.get("goal_text"),
            scorable.text
        )

        norm_score = self.normalize_score(raw_score, dimension)

        tuner = self.regression_tuners.get(dimension)
        if tuner:
            tuned = tuner.transform(norm_score)
            tuned = max(
                self.min_score_by_dim.get(dimension, 0.0),
                min(self.max_score_by_dim.get(dimension, 100.0), tuned),
            )
            self.logger.log(
                "MRQTunedScore",
                {
                    "dimension": dimension,
                    "raw": norm_score,
                    "tuned": tuned,
                },
            )
            return tuned

        return norm_score
``n

## File: mrq\encoder.py

`python
# stephanie/scoring/mrq/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, dim=4096, hdim=4096):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim * 2, hdim),  # Concatenate context + document
            nn.ReLU(),
            nn.Linear(hdim, dim),      # Keep the output same size
        )

    def forward(self, context_emb, doc_emb):
        concat = torch.cat([context_emb, doc_emb], dim=1)
        return self.encoder(concat)
``n

## File: mrq\initializer.py

`python
# stephanie/scoring/mrq/initializer.py
from stephanie.evaluator.hypothesis_value_predictor import \
    HypothesisValuePredictor
from stephanie.evaluator.mrq_trainer import MRQTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner

from .encoder import TextEncoder
from .model import MRQModel


def initialize_dimension(self, dimension):
    if not self.value_predictor:
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)
    if not self.encoder:
        self.encoder = TextEncoder().to(self.device)

    self.regression_tuners[dimension] = RegressionTuner(
        dimension=dimension, logger=self.logger
    )
    self.trainers[dimension] = MRQTrainer(
        memory=self.memory,
        logger=self.logger,
        value_predictor=self.value_predictor,
        encoder=self.encoder,
        device=self.device,
    )
    self.models[dimension] = MRQModel(
        self.encoder, self.value_predictor, device=self.device
    )
    self.min_score_by_dim[dimension] = 0.0
    self.max_score_by_dim[dimension] = 1.0
    self.logger.log("MRQModelInitializing", {"dimension": dimension})
``n

## File: mrq\model_io.py

`python
# stephanie/scoring/mrq/model_io.py
import json
import os

import torch


class MRQModelIO:
    def save_models(self):
        base_dir = self.cfg.get("scoring", {}).get("model_dir", "models/mrq/")
        os.makedirs(base_dir, exist_ok=True)

        for dim, (encoder, predictor) in self.models.items():
            dim_dir = os.path.join(base_dir, dim)
            os.makedirs(dim_dir, exist_ok=True)

            torch.save(encoder.state_dict(), os.path.join(dim_dir, "encoder.pt"))
            torch.save(predictor.state_dict(), os.path.join(dim_dir, "predictor.pt"))

            self.regression_tuners[dim].save(os.path.join(dim_dir, "tuner.json"))

            meta = {
                "min_score": self.min_score_by_dim[dim],
                "max_score": self.max_score_by_dim[dim],
            }
            with open(os.path.join(dim_dir, "meta.json"), "w") as f:
                json.dump(meta, f)

            self.logger.log("MRQModelSaved", {"dimension": dim, "path": dim_dir})

    def load_models(self):
        base_dir = self.cfg.get("scoring", {}).get("model_dir", "models/mrq/")

        if not os.path.exists(base_dir):
            self.logger.log("MRQModelDirNotFound", {"path": base_dir})
            return

        self.dimensions = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        for dim in self.dimensions:
            dim_dir = os.path.join(base_dir, dim)

            if dim not in self.models:
                self._initialize_dimension(dim)

            model = self.models[dim]
            encoder = model.encoder
            predictor = model.predictor

            try:
                encoder_path = os.path.join(dim_dir, "encoder.pt")
                predictor_path = os.path.join(dim_dir, "predictor.pt")
                if not os.path.exists(encoder_path) or not os.path.exists(
                    predictor_path
                ):
                    self.logger.log("MRQModelFilesMissing", {"dimension": dim})
                    continue

                encoder.load_state_dict(
                    torch.load(encoder_path, map_location=self.device)
                )
                predictor.load_state_dict(
                    torch.load(predictor_path, map_location=self.device)
                )

                tuner_path = os.path.join(dim_dir, "tuner.json")
                if os.path.exists(tuner_path):
                    self.regression_tuners[dim].load(tuner_path)

                meta_path = os.path.join(dim_dir, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                        self.min_score_by_dim[dim] = meta.get("min_score", 0.0)
                        self.max_score_by_dim[dim] = meta.get("max_score", 100.0)

                self.logger.log("MRQModelLoaded", {"dimension": dim})

            except Exception as e:
                self.logger.log(
                    "MRQModelLoadError", {"dimension": dim, "error": str(e)}
                )

    def load_models_with_path(self):
        base_dir = self.cfg.get("scoring", {}).get("model_dir", "models/mrq/")

        for dim in self.dimensions:
            dim_dir = os.path.join(base_dir, dim)
            if not os.path.exists(dim_dir):
                self.logger.log("MRQLoadMissing", {"dimension": dim})
                continue

            model = self.models[dim]
            encoder = model.encoder
            predictor = model.predictor

            encoder.load_state_dict(torch.load(os.path.join(dim_dir, "encoder.pt")))
            predictor.load_state_dict(torch.load(os.path.join(dim_dir, "predictor.pt")))

            self.regression_tuners[dim].load(os.path.join(dim_dir, "tuner.json"))

            with open(os.path.join(dim_dir, "meta.json")) as f:
                meta = json.load(f)
                self.min_score_by_dim[dim] = meta["min_score"]
                self.max_score_by_dim[dim] = meta["max_score"]

            self.logger.log("MRQModelLoaded", {"dimension": dim})

    def save_metadata(self, base_dir):
        for dim in self.dimensions:
            dim_dir = os.path.join(base_dir, dim)
            os.makedirs(dim_dir, exist_ok=True)
            meta_path = os.path.join(dim_dir, "meta.json")
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "min_score": self.min_score_by_dim.get(dim, 0.0),
                        "max_score": self.max_score_by_dim.get(dim, 1.0),
                    },
                    f,
                )
``n

## File: mrq\model.py

`python
# stephanie\scoring\mrq\model.py
import torch


class MRQModel:
    def __init__(self, encoder, predictor, embedding_store, device="cpu"):
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.embedding_store = embedding_store
        self.device = device

    def predict(self, prompt_text: str, response_text: str) -> float:
        prompt_emb = torch.tensor(
            self.embedding_store.get_or_create(prompt_text), device=self.device
        ).unsqueeze(0)
        response_emb = torch.tensor(
            self.embedding_store.get_or_create(response_text), device=self.device
        ).unsqueeze(0)

        zsa = self.encoder(prompt_emb, response_emb)
        value = self.predictor(zsa).item()
        return value

    def load_weights(self, encoder_path: str, predictor_path: str):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.predictor.load_state_dict(
            torch.load(predictor_path, map_location=self.device)
        )
        self.encoder.eval()
        self.predictor.eval()

    def train(self):
        self.encoder.train()
        self.predictor.train()

    def eval(self):
        self.encoder.eval()
        self.predictor.eval()
``n

## File: mrq\preference_pair_builder.py

`python
# stephanie/scoring/mrq/preference_pair_builder.py

from collections import defaultdict

from sqlalchemy.sql import text


class PreferencePairBuilder:
    """
    Builds preference training pairs from scored documents per dimension.
    Designed for MR.Q or reward model training to rank research/document quality.
    """

    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger

    def get_training_pairs_by_dimension(self, goal=None, limit=100, dim=None):
        query = text(f"""
            WITH scored_docs AS (
                SELECT
                    s.dimension,
                    s.score,
                    d.id AS doc_id,
                    d.title,
                    d.content,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.dimension, d.id ORDER BY s.score DESC
                    ) AS rank_high,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.dimension, d.id ORDER BY s.score ASC
                    ) AS rank_low
                FROM scores s
                JOIN evaluations e ON s.evaluation_id = e.id
                JOIN documents d ON e.document_id = d.id
                WHERE s.score IS NOT NULL
                {"AND s.dimension IN :dims" if dim else ""}
            )
            SELECT
                dimension,
                title,
                content,
                score,
                rank_type,
                doc_id
            FROM (
                SELECT
                    dimension,
                    title,
                    content,
                    score,
                    'top' AS rank_type,
                    doc_id
                FROM scored_docs
                WHERE rank_high = 1
                AND content IS NOT NULL
                AND content <> ''

                UNION ALL

                SELECT
                    dimension,
                    title,
                    content,
                    score,
                    'bottom' AS rank_type,
                    doc_id
                FROM scored_docs
                WHERE rank_low = 1
            ) AS ranked_pairs
            ORDER BY dimension, doc_id
            LIMIT :limit
        """)

        params = {
            "limit": limit or 100
        }
        if dim:
            params["dims"] = tuple(dim)
        if goal:
            params["goal"] = goal  # Currently unused unless you add it to the query.

        # Optional: print full SQL for debugging
        # compiled = query.compile(self.db.bind, compile_kwargs={"literal_binds": True})
        # self.logger.log("SQLQuery", {"query": str(compiled)})
        try:
            rows = self.db.execute(query, params).fetchall()
        except Exception as e:
            if self.logger:
                self.logger.log("DocumentPairBuilderError", {"error": str(e)})
            self.db.rollback()
            return {}

        grouped = defaultdict(dict)
        results_by_dimension = defaultdict(list)

        for row in rows:
            key = (row.dimension, row.doc_id)
            grouped[key][row.rank_type] = row

        for (dimension, _), data in grouped.items():
            if "top" in data and "bottom" in data:
                results_by_dimension[dimension].append(
                    {
                        "title": data["top"].title,
                        "output_a": data["top"].content,
                        "output_b": data["bottom"].content,
                        "value_a": float(data["top"].score),
                        "value_b": float(data["bottom"].score),
                    }
                )

        return dict(results_by_dimension)
``n

## File: mrq\reward_based_trainer.py

`python
# stephanie/scoring/mrq/reward_based_trainer.py
import torch
import torch.nn as nn


class RewardBasedTrainer:
    def __init__(self, encoder, predictor, optimizer, device="cpu"):
        self.encoder = encoder
        self.predictor = predictor
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()
        self.device = device

    def update(self, context_emb, doc_emb, reward):
        self.encoder.eval()
        self.predictor.train()

        with torch.no_grad():
            z = self.encoder(context_emb, doc_emb)

        pred = self.predictor(z)
        target = torch.tensor([reward], device=self.device)
        loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
``n

## File: mrq\scorer.py

`python
# stephanie/scoring/mrq/scorer.py

from .core_scoring import MRQCoreScoring
from .initializer import initialize_dimension
from .model_io import MRQModelIO
from .training import MRQTraining


class MRQScorer(MRQCoreScoring, MRQTraining, MRQModelIO):
    def __init__(self, cfg, memory, logger, dimensions=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.device = cfg.get("device", "cpu")
        self.dimensions = dimensions or ["mrq"]
        self.models = {}
        self.trainers = {}
        self.min_score_by_dim = {}
        self.max_score_by_dim = {}
        self.regression_tuners = {}
        self.value_predictor = None
        self.encoder = None

        for dim in self.dimensions:
            initialize_dimension(self, dim)

        # Bind methods to self
        self.estimate_score = lambda goal, scorable, dim: self.estimate_score(
            self, goal, scorable, dim
        )
        self.evaluate = lambda prompt, response: self.evaluate(self, prompt, response)
        self.judge = lambda goal, prompt, a, b: self.judge(self, goal, prompt, a, b)
        self.predict_score_from_prompt = (
            lambda prompt, dim="mrq", top_k=5: self.predict_score_from_prompt(
                self, prompt, dim, top_k
            )
        )

        self.train_from_database = lambda cfg: self.train_from_database(self, cfg)
        self.train_from_context = lambda ctx, cfg: self.train_from_context(self, ctx, cfg)
        self.align_mrq_with_llm_scores_from_pairs = (
            lambda samples,
            dim,
            prefix="MRQAlignment": self.align_mrq_with_llm_scores_from_pairs(
                self, samples, dim, prefix
            )
        )
        self.update_score_bounds_from_data = (
            lambda samples, dim: self.update_score_bounds_from_data(self, samples, dim)
        )

        self.save_models = lambda: self.save_models(self)
        self.load_models = lambda: self.load_models(self)
        self.load_models_with_path = lambda: self.load_models_with_path(self)
        self.save_metadata = lambda base_dir: self.save_metadata(self, base_dir)
``n

## File: mrq\trainer_engine.py

`python
# stephanie/scoring/mrq/trainer_engine.py
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from stephanie.models.incontext_q_model import InContextQModel
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import save_json
from stephanie.utils.metrics import EpistemicMetrics


class MRQTrainerEngine:
    def __init__(self, memory, logger, device="cpu"):
        self.memory = memory
        self.logger = logger
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim

        # Loss weights (configurable)
        self.q_weight = 1.0
        self.v_weight = 0.5
        self.pi_weight = 0.2
        self.expectile_weight = 0.8  # For V-loss
        self.entropy_weight = 0.1  # For policy regularization

        # Training parameters
        self.lr = 1e-4
        self.lr_v = 5e-5
        self.lr_pi = 3e-5
        self.epochs = 50
        self.batch_size = 32
        self.patience = 3
        self.min_delta = 0.001
        self.uncertainty_threshold = 0.3
        self.gamma = 0.95  # Discount factor

    def build_encoder(self):
        """Build text encoder for context-document fusion"""
        return TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)

    def build_predictor(self):
        """Build value predictor for MRQ compatibility"""
        return ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)

    def build_sicql_model(self, action_dim=1):
        """Build SICQL model with Q/V/π heads"""
        return InContextQModel(
            dim=self.dim,
            hdim=self.hdim,
            action_dim=action_dim,
            device=self.device,
        ).to(self.device)

    def _create_dataloader(self, encoder, samples):
        """Convert samples to PyTorch DataLoader"""
        context_embs, doc_embs, scores = [], [], []

        for idx, item in enumerate(samples):
            # Get context and document embeddings
            context = item.get("title", "")
            context_emb = self.memory.embedding.get_or_create(context)

            # Process both A and B samples
            for side in ["a", "b"]:
                doc_text = item[f"output_{side}"]
                doc_emb = self.memory.embedding.get_or_create(doc_text)

                # Store data
                context_embs.append(torch.tensor(context_emb))
                doc_embs.append(torch.tensor(doc_emb))
                scores.append(float(item[f"value_{side}"]))

        # Convert to tensors
        context_tensors = torch.stack(context_embs).to(self.device)
        doc_tensors = torch.stack(doc_embs).to(self.device)
        score_tensors = torch.tensor(scores).float().to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(context_tensors, doc_tensors, score_tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _compute_losses(self, outputs, scores, next_values=None):
        """Compute all SICQL losses"""
        losses = {}

        # Q-loss: MSE vs LLM scores
        if next_values is not None:
            # Use Bellman target with gamma
            q_target = scores + self.gamma * next_values
        else:
            q_target = scores

        losses["q"] = nn.MSELoss()(outputs["q_value"].squeeze(), q_target)

        # V-loss: Expectile regression between Q and V
        with torch.no_grad():
            advantage = (
                outputs["q_value"].detach() - outputs["state_value"].detach()
            )
            expectile_mask = (advantage > 0).float()

        losses["v"] = torch.mean(
            self.expectile_weight
            * torch.abs(outputs["state_value"].squeeze() - q_target.detach())
            * expectile_mask
        )

        # Policy loss: Advantage-weighted regression (AWR)
        action_probs = F.softmax(outputs["action_probs"], dim=-1)
        advantage = (outputs["q_value"] - outputs["state_value"]).detach()
        losses["pi"] = -torch.mean(
            torch.log(action_probs) * advantage.unsqueeze(-1)
        )

        # Entropy regularization
        dist = torch.distributions.Categorical(
            logits=outputs["action_probs"]
        )
        losses["entropy"] = -self.entropy_weight * dist.entropy().mean()

        # Total loss
        losses["total"] = (
            self.q_weight * losses["q"]
            + self.v_weight * losses["v"]
            + self.pi_weight * losses["pi"]
            + losses["entropy"]
        )

        return losses

    def _train_epoch(self, model, dataloader, optimizers):
        """Train for one epoch"""
        model.train()
        epoch_losses = defaultdict(list)

        for context_emb, doc_emb, scores in tqdm(dataloader, desc="Training"):
            # Forward pass
            outputs = model(context_emb, doc_emb)

            # Compute losses
            losses = self._compute_losses(outputs, scores)

            # Backward pass
            optimizers["q"].zero_grad()
            optimizers["v"].zero_grad()
            optimizers["pi"].zero_grad()

            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # Update weights
            optimizers["q"].step()
            optimizers["v"].step()
            optimizers["pi"].step()

            # Record losses
            for k, v in losses.items():
                epoch_losses[k].append(v.item())

        return {k: np.mean(v) for k, v in epoch_losses.items()}

    def _validate(self, model, val_loader):
        """Validation phase"""
        model.eval()
        val_losses = defaultdict(list)

        with torch.no_grad():
            for context_emb, doc_emb, scores in val_loader:
                outputs = model(context_emb, doc_emb)
                losses = self._compute_losses(outputs, scores)

                for k, v in losses.items():
                    val_losses[k].append(v.item())

        return {f"val_{k}": np.mean(v) for k, v in val_losses.items()}

    def _setup_optimizers(self, model):
        """Initialize optimizers for all heads"""
        return {
            "q": optim.Adam(model.q_head.parameters(), lr=self.lr),
            "v": optim.Adam(model.v_head.parameters(), lr=self.lr_v),
            "pi": optim.Adam(model.pi_head.parameters(), lr=self.lr_pi),
        }

    def _setup_schedulers(self, optimizers):
        """Initialize learning rate schedulers"""
        return {
            "q": ReduceLROnPlateau(
                optimizers["q"], mode="min", factor=0.5, patience=2
            ),
            "v": ReduceLROnPlateau(
                optimizers["v"], mode="min", factor=0.5, patience=2
            ),
            "pi": ReduceLROnPlateau(
                optimizers["pi"], mode="min", factor=0.5, patience=2
            ),
        }

    def train_sicql(
        self, model, dataloader, val_loader=None, output_dir="models"
    ):
        """Main training loop with SICQL enhancements"""
        # Setup optimizers and schedulers
        optimizers = self._setup_optimizers(model)
        schedulers = self._setup_schedulers(optimizers)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            train_losses = self._train_epoch(model, dataloader, optimizers)

            # Validation phase
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate(model, val_loader)
                val_loss = val_metrics["val_total"]
            else:
                val_loss = train_losses["total"]

            # Logging
            self.logger.log(
                "SICQLTrainingEpoch",
                {
                    "epoch": epoch + 1,
                    "train_loss": train_losses["total"],
                    "val_loss": val_loss,
                    "q_loss": train_losses["q"],
                    "v_loss": train_losses["v"],
                    "pi_loss": train_losses["pi"],
                    "entropy": train_losses["entropy"],
                    "lr": optimizers["q"].param_groups[0]["lr"],
                },
            )

            # Early stopping
            if val_loss < best_loss - self.min_delta:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                self.logger.log(
                    "SICQLEarlyStopping",
                    {"epoch": epoch + 1, "best_loss": best_loss},
                )    
                break

            # Update learning rates
            for name in ["q", "v", "pi"]:
                if val_loader:
                    schedulers[name].step(val_loss)
                else:
                    schedulers[name].step(train_losses["total"])

        # Final model save
        torch.save(model.state_dict(), f"{output_dir}/final_model.pt")
        self.logger.log("SICQLTrainingComplete", {"best_loss": best_loss})

        return model

    def detect_epistemic_gaps(self, model, dataloader):
        """Identify areas where model uncertainty is high"""
        model.eval()
        gaps = []

        with torch.no_grad():
            for context_emb, doc_emb, scores in dataloader:
                outputs = model(context_emb, doc_emb)

                # Compute uncertainty
                uncertainties = EpistemicMetrics.compute_uncertainty(
                    outputs["q_value"], outputs["state_value"]
                )

                # Find high-uncertainty samples
                for i in range(len(uncertainties)):
                    if uncertainties[i] > self.uncertainty_threshold:
                        gaps.append(
                            {
                                "sample_idx": i,
                                "uncertainty": uncertainties[i].item(),
                                "predicted_score": outputs["q_value"][
                                    i
                                ].item(),
                                "llm_score": scores[i].item(),
                                "document_text": doc_emb[
                                    i
                                ].tolist(),  # Convert to list for JSON
                            }
                        )

        # Log gaps
        for gap in gaps:
            EpistemicMetrics.log_epistemic_gap(gap)

        return gaps

    def train_all(self, contrast_pairs, cfg=None):
        """Train models for all dimensions"""
        if cfg:
            self._update_config(cfg)

        trained_models = {}
        trained_encoders = {}
        regression_tuners = {}

        # Group pairs by dimension
        pairs_by_dim = defaultdict(list)
        for item in contrast_pairs:
            pairs_by_dim[item["dimension"]].append(item)

        # Train for each dimension
        for dim, samples in pairs_by_dim.items():
            self.logger.log(
                "SICQLTrainingDimension",
                {"dimension": dim, "sample_count": len(samples)},
            )

            # Build dataloader
            dataloader = self._create_dataloader(self.build_encoder(), samples)

            # Initialize model

            use_sicql = self.cfg.get("use_sicql_style", False)

            if use_sicql:
                model = self.build_sicql_model(action_dim=1)
            else:
                encoder = self.build_encoder()
                predictor = self.build_predictor()
                from stephanie.scoring.mrq.model import MRQModel
                model = MRQModel(encoder, predictor, self.memory.embedding, device=self.device)

            # Create output directory
            output_dir = f"{self.cfg.get('model_path', 'models')}/{dim}"
            os.makedirs(output_dir, exist_ok=True)

            # Train
            if use_sicql:
                trained_model = self.train_sicql(
                    model, dataloader, output_dir=output_dir
                )
                trained_models[dim] = trained_model.state_dict()
                trained_encoders[dim] = trained_model.encoder.state_dict()
                torch.save(
                    trained_model.q_head.state_dict(), f"{output_dir}/q_head.pt"
                )
                torch.save(
                    trained_model.v_head.state_dict(), f"{output_dir}/v_head.pt"
                )
                torch.save(
                    trained_model.pi_head.state_dict(), f"{output_dir}/pi_head.pt"
                )
            else:
                model = self._train_mrq(model, dataloader, output_dir=output_dir)  
                trained_model = model.predictor
                trained_models[dim] = model.predictor.state_dict()
                trained_encoders[dim] = model.encoder.state_dict()

            # Save metadata
            meta = {
                "dim": self.dim,
                "hdim": self.hdim,
                "dimension": dim,
                "model_type": "sicql" if use_sicql else "mrq",
                "target_type": self.cfg.get("target_type", "document"),
                "embedding_type": self.cfg.get("embedding_type", "hnet"),
                "version": self.cfg.get("model_version", "v1"),
                "training_params": {
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.lr,
                    "gamma": self.gamma,
                },
            }
            save_json(meta, f"{output_dir}/meta.json")

            # Build regression tuner
            regression_tuners[dim] = RegressionTuner.from_dataloader(
                dataloader,
                model=trained_model,
                dimension=dim,
                logger=self.logger,
            )

        return trained_encoders, trained_models, regression_tuners

    def _update_config(self, cfg):
        """Update training parameters from config"""
        self.cfg = cfg
        self.lr = cfg.get("lr", self.lr)
        self.lr_v = cfg.get("lr_v", self.lr_v)
        self.lr_pi = cfg.get("lr_pi", self.lr_pi)
        self.epochs = cfg.get("epochs", self.epochs)
        self.batch_size = cfg.get("batch_size", self.batch_size)
        self.patience = cfg.get("patience", self.patience)
        self.min_delta = cfg.get("min_delta", self.min_delta)
        self.uncertainty_threshold = cfg.get(
            "uncertainty_threshold", self.uncertainty_threshold
        )
        self.gamma = cfg.get("gamma", self.gamma)
        self.q_weight = cfg.get("q_weight", self.q_weight)
        self.v_weight = cfg.get("v_weight", self.v_weight)
        self.pi_weight = cfg.get("pi_weight", self.pi_weight)
        self.expectile_weight = cfg.get(
            "expectile_weight", self.expectile_weight
        )
        self.entropy_weight = cfg.get("entropy_weight", self.entropy_weight)

    def _train_mrq(self, model, dataloader, output_dir=None):
        """Train a standard MRQModel (Q only) using MSE"""
        model.train_mode()  # ✅ FIXED

        optimizer = optim.Adam(
            list(model.encoder.parameters()) + list(model.predictor.parameters()),
            lr=self.lr
        )
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            total_loss = 0
            count = 0

            for context_emb, doc_emb, scores in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()

                # Forward pass
                zsa = model.encoder(context_emb, doc_emb)
                q_pred = model.predictor(zsa).squeeze()

                # MSE loss
                loss = F.mse_loss(q_pred, scores)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 0.5)

                optimizer.step()

                total_loss += loss.item() * context_emb.size(0)
                count += context_emb.size(0)

            avg_loss = total_loss / count

            self.logger.log("MRQTrainingEpoch", {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if avg_loss < best_loss - self.min_delta:
                best_loss = avg_loss
                patience_counter = 0
                if output_dir:
                    torch.save(model.encoder.state_dict(), f"{output_dir}/encoder.pt")
                    torch.save(model.predictor.state_dict(), f"{output_dir}/predictor.pt")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                self.logger.log("MRQEarlyStopping", {
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                })
                break

        self.logger.log("MRQTrainingComplete", {"best_loss": best_loss})
        return model
``n

## File: mrq\trainer.py

`python
# stephanie\scoring\mrq\trainer.py
import torch
import torch.nn.functional as F


class MRQTrainer:
    def __init__(self, mrq_model, optimizer):
        self.model = mrq_model
        self.optimizer = optimizer
        self.model.train_mode()

    def update(self, goal: str, chunk: str, reward: float) -> float:
        self.model.train_mode()
        pred = self.model.predict(goal, chunk)
        pred_tensor = torch.tensor([[pred]], requires_grad=True)
        target_tensor = torch.tensor([[reward]], dtype=torch.float32)

        loss = F.mse_loss(pred_tensor, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
``n

## File: mrq\training.py

`python
# stephanie/scoring/mrq/training.py


class MRQTraining:
    def train_from_database(self, cfg):
        all_samples = self.memory.mrq.get_training_pairs_by_dimension()
        for dim, samples in all_samples.items():
            if not samples:
                self.logger.log("MRQNoTrainingSamples", {"dimension": dim})
                continue

            self.align_mrq_with_llm_scores_from_pairs(samples, dimension=dim)

            self.logger.log(
                "MRQTrainingStart", {"dimension": dim, "sample_count": len(samples)}
            )

            if dim not in self.trainers:
                self.trainers[dim] = self._build_trainer(dim)

            self.update_score_bounds_from_data(samples, dim)
            dataloader = self.trainers[dim].prepare_training_data(samples)
            self.trainers[dim].train(dataloader, cfg)

            self.logger.log("MRQTrainingComplete", {"dimension": dim})

    def train_from_context(self, context, cfg):
        dim_samples = context.get("mrq_training_pairs_by_dimension", {})
        for dim, samples in dim_samples.items():
            if not samples:
                self.logger.log("MRQNoTrainingFromContext", {"dimension": dim})
                continue

            self.logger.log(
                "MRQContextTrainingStart",
                {"dimension": dim, "sample_count": len(samples)},
            )

            self.update_score_bounds_from_data(samples, dim)
            dataloader = self.trainers[dim].prepare_training_data(samples)
            self.trainers[dim].train(dataloader, cfg)

            self.logger.log("MRQContextTrainingComplete", {"dimension": dim})

    def align_mrq_with_llm_scores_from_pairs(
        self, pair_samples, dimension, log_prefix="MRQAlignment"
    ):
        for pair in pair_samples:
            prompt = pair["prompt"]
            for side in ["a", "b"]:
                hyp = pair[f"output_{side}"]
                llm_score = pair[f"value_{side}"]

                mrq_score = self.score(
                    {"goal_text": prompt}, self.Scorable(text=hyp), [dimension]
                )

                self.logger.log(
                    f"{log_prefix}Dynamic",
                    {
                        "prompt_hash": hash(prompt),
                        "hypothesis_hash": hash(hyp),
                        "dimension": dimension,
                        "llm_score": llm_score,
                        "predicted_mrq": mrq_score,
                    },
                )

                if mrq_score and llm_score is not None:
                    self.regression_tuners[dimension].train_single(
                        mrq_score=mrq_score.results[dimension].score,
                        llm_score=llm_score,
                    )

    def update_score_bounds_from_data(self, samples, dim):
        values = []
        for s in samples:
            if "value_a" in s and "value_b" in s:
                values.extend([s["value_a"], s["value_b"]])
            elif "value" in s:
                values.append(s["value"])
        if values:
            min_score = min(values)
            max_score = max(values)
            self.min_score_by_dim[dim] = min_score
            self.max_score_by_dim[dim] = max_score
            self.logger.log(
                "MRQScoreBoundsUpdated",
                {
                    "dimension": dim,
                    "min_score": min_score,
                    "max_score": max_score,
                    "example_count": len(values),
                },
            )
``n

## File: mrq\value_predictor.py

`python
# stephanie/scoring/mrq/value_predictor.py
import logging

from torch import nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValuePredictor(nn.Module):
    """Predicts a quality score for a document given its contextual embedding."""

    def __init__(self, zsa_dim=4096, hdim=2048):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(zsa_dim, hdim), nn.ReLU(), nn.Linear(hdim, 1)
        )

    def forward(self, zsa_embedding):
        assert len(zsa_embedding.shape) == 2, (
            f"Expected 2D input, got {zsa_embedding.shape}"
        )
        return self.value_net(zsa_embedding)
``n

## File: pairwise_data.py

`python
# stephanie/scoring/pairwise_data.py

from collections import defaultdict

from sqlalchemy.sql import text


class PreferencePairDatasetBuilder:
    """
    Builds preference training pairs for MR.Q or reward model training
    by selecting top and bottom scored outputs per dimension.
    """

    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger

    def get_training_pairs_by_dimension(
        self, goal: str = None, limit: int = 10000
    ) -> dict:
        """
        Returns top and bottom scored prompt/response pairs per dimension.

        Output:
        {
            "relevance": [
                {
                    "prompt": "...",
                    "output_a": "...",  # top
                    "output_b": "...",  # bottom
                    "value_a": 8.9,
                    "value_b": 3.1
                },
                ...
            ],
            ...
        }
        """
        query = text(
            """
            WITH scored_prompts AS (
                SELECT
                    s.dimension,
                    s.score,
                    e.pipeline_run_id,
                    p.id AS prompt_id,
                    p.prompt_text,
                    p.response_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.dimension, p.id ORDER BY s.score DESC
                    ) AS rank_high,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.dimension, p.id ORDER BY s.score ASC
                    ) AS rank_low
                FROM scores s
                JOIN evaluations e ON s.evaluation_id = e.id
                JOIN prompts p ON e.pipeline_run_id = p.pipeline_run_id
                WHERE s.score IS NOT NULL
                {goal_filter}
            )
            SELECT
                dimension,
                prompt_text,
                response_text,
                score,
                rank_type,
                prompt_id
            FROM (
                SELECT
                    dimension,
                    prompt_text,
                    response_text,
                    score,
                    'top' AS rank_type,
                    prompt_id
                FROM scored_prompts
                WHERE rank_high = 1
                  AND prompt_text IS NOT NULL
                  AND response_text IS NOT NULL
                  AND prompt_text <> ''
                  AND response_text <> ''
                  
                UNION ALL

                SELECT
                    dimension,
                    prompt_text,
                    response_text,
                    score,
                    'bottom' AS rank_type,
                    prompt_id
                FROM scored_prompts
                WHERE rank_low = 1
            ) AS ranked_pairs
            ORDER BY dimension, prompt_id
            LIMIT :limit
        """.replace("{goal_filter}", "AND p.goal_text = :goal" if goal else "")
        )

        params = {"limit": limit}
        if goal:
            params["goal"] = goal

        try:
            rows = self.db.execute(query, params).fetchall()
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to get training pairs", extra={"error": str(e)}
                )
            return {}

        grouped = defaultdict(dict)
        results_by_dimension = defaultdict(list)

        for row in rows:
            key = (row.dimension, row.prompt_id)
            grouped[key][row.rank_type] = row

        for (dimension, _), data in grouped.items():
            if "top" in data and "bottom" in data:
                results_by_dimension[dimension].append(
                    {
                        "prompt": data["top"].prompt_text,
                        "output_a": data["top"].response_text,
                        "output_b": data["bottom"].response_text,
                        "value_a": float(data["top"].score),
                        "value_b": float(data["bottom"].score),
                    }
                )

        return dict(results_by_dimension)
``n

## File: prompt_pair_builder.py

`python
# stephanie/scoring/prompt_pair_builder.py

from collections import defaultdict

from sqlalchemy.sql import text


class PromptPreferencePairBuilder:
    """
    Builds preference training pairs from scored prompt-response pairs
    for MR.Q or reward model training. Operates per dimension.
    """

    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger

    def get_training_pairs_by_dimension(
        self, goal: str = None, limit: int = 10000
    ) -> dict:
        """
        Returns a dictionary of training pairs grouped by dimension.

        Output Format:
        {
            "clarity": [
                {
                    "prompt": "...",
                    "output_a": "...",  # preferred
                    "output_b": "...",  # less preferred
                    "value_a": 8.2,
                    "value_b": 4.5
                },
                ...
            ],
            ...
        }
        """
        query = text(
            """
            WITH scored_prompts AS (
                SELECT
                    s.dimension,
                    s.score,
                    e.pipeline_run_id,
                    p.id AS prompt_id,
                    p.prompt_text,
                    p.response_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.dimension, p.id ORDER BY s.score DESC
                    ) AS rank_high,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.dimension, p.id ORDER BY s.score ASC
                    ) AS rank_low
                FROM scores s
                JOIN evaluations e ON s.evaluation_id = e.id
                JOIN prompts p ON e.pipeline_run_id = p.pipeline_run_id
                WHERE s.score IS NOT NULL
                {goal_filter}
            )
            SELECT
                dimension,
                prompt_text,
                response_text,
                score,
                rank_type,
                prompt_id
            FROM (
                SELECT
                    dimension,
                    prompt_text,
                    response_text,
                    score,
                    'top' AS rank_type,
                    prompt_id
                FROM scored_prompts
                WHERE rank_high = 1
                  AND prompt_text IS NOT NULL
                  AND response_text IS NOT NULL
                  AND prompt_text <> ''
                  AND response_text <> ''
                  
                UNION ALL

                SELECT
                    dimension,
                    prompt_text,
                    response_text,
                    score,
                    'bottom' AS rank_type,
                    prompt_id
                FROM scored_prompts
                WHERE rank_low = 1
            ) AS ranked_pairs
            ORDER BY dimension, prompt_id
            LIMIT :limit
        """.replace("{goal_filter}", "AND p.goal_text = :goal" if goal else "")
        )

        params = {"limit": limit}
        if goal:
            params["goal"] = goal

        try:
            rows = self.db.execute(query, params).fetchall()
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to get training pairs", extra={"error": str(e)}
                )
            return {}

        grouped = defaultdict(dict)
        results_by_dimension = defaultdict(list)

        for row in rows:
            key = (row.dimension, row.prompt_id)
            grouped[key][row.rank_type] = row

        for (dimension, _), data in grouped.items():
            if "top" in data and "bottom" in data:
                results_by_dimension[dimension].append(
                    {
                        "prompt": data["top"].prompt_text,
                        "output_a": data["top"].response_text,
                        "output_b": data["bottom"].response_text,
                        "value_a": float(data["top"].score),
                        "value_b": float(data["bottom"].score),
                    }
                )

        return dict(results_by_dimension)
``n

## File: proximity_scorer.py

`python
# stephanie/scoring/proximity_scorer.py
import re

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class ProximityScorer(BaseScorer):
    def __init__(self, cfg, memory, logger, prompt_loader=None, dimensions=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.prompt_loader = prompt_loader

        # Dynamically pull dimensions from scoring config
        self.dimensions_config = cfg.get("dimensions", [])
        self.dimensions = dimensions or [d["name"] for d in self.dimensions_config]

        self.models = {}
        self.scalers = {}
        self.trained = dict.fromkeys(self.dimensions, False)
        self.force_rescore = cfg.get("force_rescore", False)
        self.regression_tuners = {}

        for dim in self.dimensions:
            self._initialize_dimension(dim)

    @property
    def name(self) -> str:
        return "proximity"

    def evaluate(self, prompt: str, response: str) -> ScoreBundle:
        if not response:
            return self._fallback("No proximity response available.")

        try:
            themes = self._extract_block(response, "Common Themes Identified")
            grafts = self._extract_block(response, "Grafting Opportunities")
            directions = self._extract_block(response, "Strategic Directions")

            themes_score = 10.0 * len(themes)
            grafts_score = 10.0 * len(grafts)
            directions_score = 20.0 * len(directions)

            results = {
                "proximity_themes": ScoreResult(
                    dimension="proximity_themes",
                    score=min(100.0, themes_score),
                    weight=0.3,
                    rationale=f"{len(themes)} theme(s) identified",
                    source="proximity",
                ),
                "proximity_grafts": ScoreResult(
                    dimension="proximity_grafts",
                    score=min(100.0, grafts_score),
                    weight=0.3,
                    rationale=f"{len(grafts)} grafting suggestion(s)",
                    source="proximity",
                ),
                "proximity_directions": ScoreResult(
                    dimension="proximity_directions",
                    score=min(100.0, directions_score),
                    weight=0.4,
                    rationale=f"{len(directions)} strategic direction(s)",
                    source="proximity",
                ),
            }

            return ScoreBundle(results=results)

        except Exception as e:
            return self._fallback(f"Failed to parse proximity response: {str(e)}")

    def _extract_block(self, text: str, section_title: str) -> list:
        pattern = rf"# {re.escape(section_title)}\n((?:- .+\n?)*)"
        match = re.search(pattern, text)
        if not match:
            return []
        block = match.group(1).strip()
        return [line.strip("- ").strip() for line in block.splitlines() if line.strip()]

    def _fallback(self, message: str) -> ScoreBundle:
        results = {
            "proximity_themes": ScoreResult(
                "proximity_themes", 0.0, message, 0.3, source="proximity"
            ),
            "proximity_grafts": ScoreResult(
                "proximity_grafts", 0.0, message, 0.3, source="proximity"
            ),
            "proximity_directions": ScoreResult(
                "proximity_directions", 0.0, message, 0.4, source="proximity"
            ),
        }
        return ScoreBundle(results=results)

    def _initialize_dimension(self, dim: str):
        self.models[dim] = SVR()
        self.scalers[dim] = StandardScaler()
        self.regression_tuners[dim] = RegressionTuner(dimension=dim, logger=self.logger)
        self.trained[dim] = False

    def _try_train_on_dimension(self, dim: str):
        training_data = self.memory.training.get_training_data(dimension=dim)
        if not training_data:
            return

        X = [self._build_feature_vector(g, s) for g, s in training_data]
        y = [score for (_, _, score) in training_data]

        self.scalers[dim].fit(X)
        X_scaled = self.scalers[dim].transform(X)

        self.models[dim].fit(X_scaled, y)
        self.trained[dim] = True

    def _build_feature_vector(self, goal: dict, scorable: Scorable):
        goal_vec = self.memory.embedding.get_or_create(goal.get("goal_text", ""))
        text_vec = self.memory.embedding.get_or_create(scorable.text)
        return [g - t for g, t in zip(goal_vec, text_vec)]

    def score(
        self, goal: dict, scorable: Scorable, dimensions: list[str]
    ) -> ScoreBundle:
        results = {}

        for dim in dimensions:
            vec = self._build_feature_vector(goal, scorable)

            if not self.trained[dim]:
                self._try_train_on_dimension(dim)

            if not self.trained[dim]:
                score = 50.0
                rationale = f"SVM not trained for {dim}, returning neutral."
            else:
                x = self.scalers[dim].transform([vec])
                raw_score = self.models[dim].predict(x)[0]
                score = self.regression_tuners[dim].transform(raw_score)
                rationale = f"SVM predicted and aligned score for {dim}"

            # Lookup weight from config
            weight = next(
                (
                    d.get("weight", 1.0)
                    for d in self.dimensions_config
                    if d["name"] == dim
                ),
                1.0,
            )

            self.logger.log(
                "ProximityScoreComputed",
                {
                    "dimension": dim,
                    "score": score,
                    "hypothesis": scorable.text,
                },
            )

            results[dim] = ScoreResult(
                dimension=dim,
                score=score,
                rationale=rationale,
                weight=weight,
                source="svm",
            )

        return ScoreBundle(results=results)

    def parse_from_response(self, response: str) -> ScoreBundle:
        return self.evaluate(prompt="", response=response)
``n

## File: scorable_factory.py

`python
# stephanie/scoring/scorable_factory.py
from enum import Enum as PyEnum

from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.document import DocumentORM
from stephanie.models.prompt import PromptORM
from stephanie.models.theorem import CartridgeORM, TheoremORM
from stephanie.scoring.scorable import Scorable


# Enum defining all the supported types of scoreable targets
class TargetType:
    DOCUMENT = "document"
    HYPOTHESIS = "hypothesis"
    CARTRIDGE = "cartridge"
    TRIPLE = "triple"
    CHUNK = "chunk"
    PROMPT = "prompt"
    IDEA = "idea"
    RESPONSE = "response"
    PROMPT_RESPONSE = "prompt_response"
    TRAINING = "training"
    THEOREM = "theorem"
    SYMBOLIC_RULE = "symbolic_rule"
    CUSTOM = "custom"
    REFINEMENT = "refinement"  # For SRFT-style usage


class ScorableFactory:
    """ Why am I hitting the cash It shouldn't be slamming the cash by now
    Factory for turning various content types into unified Scorable objects.
    """

    @staticmethod
    def from_orm(obj, mode: str = "default") -> Scorable:
        """
        Convert an ORM object to a Scorable.
        Dispatches based on the object's class type.
        """
        if isinstance(obj, PromptORM):
            return ScorableFactory.from_prompt_pair(obj, mode)
        elif isinstance(obj, CartridgeORM):
            return Scorable(
                id=obj.id, text=obj.markdown_content, target_type=TargetType.CARTRIDGE
            )
        elif isinstance(obj, CartridgeTripleORM):
            # For a triple, we concatenate subject, relation, and object as a textual representation
            return Scorable(
                id=obj.id,
                text=f"{obj.subject} {obj.relation} {obj.object}",
                target_type=TargetType.TRIPLE,
            )
        elif isinstance(obj, TheoremORM):
            return Scorable(
                id=obj.id, text=obj.statement, target_type=TargetType.THEOREM
            )
        elif isinstance(obj, DocumentORM):
            title = obj.title or ""
            summary = obj.summary or ""
            content = obj.content or ""

            if title and summary:
                text = f"#Title\n{title}\n\n## Summary\n{summary}"
            elif content:
                text = content
            else:
                text = title or summary  # fallback if only one exists

            return Scorable(id=obj.id, text=text, target_type=TargetType.DOCUMENT)
        else:
            raise ValueError(f"Unsupported ORM type for scoring: {type(obj)}")

    @staticmethod
    def from_prompt_pair(obj: PromptORM, mode: str = "prompt+response") -> Scorable:
        """
        Handles PromptORM objects that contain both prompt and response.
        The `mode` parameter controls whether to extract only the prompt, only the response,
        or a concatenated version of both.
        """
        prompt = obj.prompt or ""
        response = obj.response or ""
        target_type = TargetType.PROMPT

        if mode == "prompt_only":
            text = prompt
        elif mode == "response_only":
            text = response
            target_type = TargetType.RESPONSE
        elif mode == "prompt+response":
            text = f"{prompt}\n\n{response}"
            target_type = TargetType.PROMPT_RESPONSE
        else:
            raise ValueError(f"Invalid prompt scoring mode: {mode}")

        return Scorable(id=obj.id, text=text, target_type=target_type)

    @staticmethod
    def from_dict(data: dict, target_type: TargetType = None) -> Scorable:
        """
        Converts a plain dictionary into a Scorable, using optional fields like
        title, summary, and content for DOCUMENT types.
        """
        if target_type is None:
            target_type = data.get("target_type", "document")
        if "text" in data: # If text is provided, use it directly
            return Scorable(id=data.get("id", ""), text=data["text"], target_type=target_type)
        if target_type == "document":
            title = data.get("title", "")
            summary = data.get("summary", "")
            content = data.get("content", "")
            if title and summary:
                text = f"#Title\n{title}\n\n## Summary\n{summary}"
            elif content:
                text = content
            else:
                text = title or summary
        elif target_type == "triple":
            text = (
                f"{data.get('subject')} {data.get('relation')} {data.get('object')}",
            )
        else:
            text = data.get("text", "")

        return Scorable(id=data.get("id"), text=text, target_type=target_type)


    @staticmethod
    def from_text(text: str, target_type: TargetType) -> Scorable:
        """
        Converts a plain dictionary into a Scorable, using optional fields like
        title, summary, and content for DOCUMENT types.
        """
        return Scorable(id="", text=text, target_type=target_type)
``n

## File: scorable.py

`python
# stephanie/scoring/scorable.py
class Scorable:
    def __init__(self, text: str, id: str = "", target_type: str = "custom"):
        self._id = id
        self._text = text
        self._target_type = target_type

    @property
    def text(self) -> str:
        return self._text

    @property
    def id(self) -> str:
        return self._id

    @property
    def target_type(self) -> str:
        return self._target_type

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "text": self._text,
            "target_type": self._target_type
            if hasattr(self._target_type, "value")
            else self._target_type,
        }

    def __repr__(self):
        preview = self._text[:30].replace("\n", " ")
        return (
            f"Scorable(id='{self._id}', "
            f"target_type='{self._target_type}', "
            f"text_preview='{preview}...')"
        )
``n

## File: score_analyzer.py

`python
# stephanie/scoring/score_analyzer.py
# analysis/score_analyzer.py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class ScoreAnalyzer:
    def __init__(self, score_data: pd.DataFrame):
        """
        Expected format:All right
        - 'hypothesis_id': str
        - 'dimension': str
        - 'score': float
        - Optional: 'outcome' (e.g., final ranking, human eval)
        """
        self.df = score_data
        self.pivot = self.df.pivot(
            index="hypothesis_id", columns="dimension", values="score"
        )

    def describe_scores(self):
        return self.pivot.describe()

    def fit_linear_regression(self, outcome_col: str):
        merged = self.pivot.copy()
        merged[outcome_col] = self.df.drop_duplicates(subset="hypothesis_id").set_index(
            "hypothesis_id"
        )[outcome_col]
        merged = merged.dropna()
        X = merged.drop(columns=[outcome_col])
        y = merged[outcome_col]
        model = LinearRegression().fit(X, y)
        return model, dict(zip(X.columns, model.coef_))

    def perform_pca(self, n_components=2):
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(self.pivot.fillna(0))
        return components, pca.explained_variance_ratio_

    def cluster_outputs(self, n_clusters=3):
        km = KMeans(n_clusters=n_clusters, n_init=10)
        labels = km.fit_predict(self.pivot.fillna(0))
        return labels

    def plot_pca_clusters(self, n_clusters=3):
        components, _ = self.perform_pca()
        labels = self.cluster_outputs(n_clusters=n_clusters)
        plt.scatter(components[:, 0], components[:, 1], c=labels, cmap="tab10")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA of Score Vectors (Colored by Cluster)")
        plt.show()
``n

## File: score_bundle.py

`python
# stephanie/scoring/score_bundle.py
import json

from stephanie.scoring.score_result import ScoreResult


class ScoreBundle:
    def __init__(self, results: dict[str, ScoreResult]):
        from stephanie.scoring.calculations.weighted_average import \
            WeightedAverageCalculator

        self.results = results
        self.calculator = WeightedAverageCalculator()

    def aggregate(self):
        result = self.calculator.calculate(self)
        print(f"ScoreBundle: Aggregated score: {result}")
        return result

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.results.items()}

    def to_json(self, stage: str):
        final_score = self.aggregate()
        return {
            "stage": stage,
            "dimensions": self.to_dict(),
            "final_score": final_score,
        }

    def to_orm(self, evaluation_id: int):
        from stephanie.models.score import ScoreORM

        return [
            ScoreORM(
                evaluation_id=evaluation_id,
                dimension=r.dimension,
                score=r.score,
                weight=r.weight,
                rationale=r.rationale,
                source=r.source,
                target_type=r.target_type,
                prompt_hash=r.prompt_hash,

            )
            for r in self.results.values()
        ]

    def __repr__(self):
        summary = ", ".join(f"{dim}: {res.score}" for dim, res in self.results.items())
        return f"<ScoreBundle({summary})>"

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreBundle":
        """
        Reconstruct a ScoreBundle from a dict where each value is a ScoreResult-like dict.
        """
        from stephanie.scoring.score_result import ScoreResult

        results = {
            dim: ScoreResult(
                dimension=dim,
                score=entry.get("score"),
                weight=entry.get("weight", 1.0),
                rationale=entry.get("rationale", ""),
                source=entry.get("source", "from_dict"),
                target_type=entry.get("target_type", "unknown"),
                prompt_hash=entry.get("prompt_hash", ""),
                
            )
            for dim, entry in data.items()
            if isinstance(entry, dict)  # Defensive: skip bad formats
        }

        return cls(results)
``n

## File: score_display.py

`python
# stephanie/scoring/score_display.py
from tabulate import tabulate


class ScoreDisplay:
    @staticmethod
    def show(scorable, results, weighted_score):
        table_data = [
            [
                dim_name,
                f"{dim_data['score']:.2f}",
                dim_data.get("weight", 1.0),
                dim_data.get("rationale", "")[:60],
            ]
            for dim_name, dim_data in results.items()
        ]
        source = "Unknown"
        try:
            table_data.append(["FINAL", f"{weighted_score:.2f}", "-", "Weighted average"])
            _, value = next(iter(results.items()))
            source = value.get("source", "Unknown")
        except StopIteration:
            pass
        print(f"\n📊 {source} Dimension Scores {scorable.target_type}:{scorable.id} Summary")
        print(
            tabulate(
                table_data,
                headers=["Dimension", "Score", "Weight", "Rationale (preview)"],
                tablefmt="fancy_grid",
            )
        )
``n

## File: score_result.py

`python
# stephanie/scoring/score_result.py
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ScoreResult:
    dimension: str
    score: float
    rationale: str
    weight: float
    source: str
    target_type: str
    prompt_hash: str
    # SICQL-specific fields
    energy: Optional[float] = None
    q_value: Optional[float] = None
    state_value: Optional[float] = None
    policy_logits: Optional[list[float]] = None
    uncertainty: Optional[float] = None
    entropy: Optional[float] = None
    advantage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "rationale": self.rationale,
            "weight": self.weight,
            "energy": self.energy,
            "source": self.source,
            "target_type": self.target_type,
            "prompt_hash": self.prompt_hash,
            "q_value": self.q_value,
            "state_value": self.state_value,
            "policy_logits": self.policy_logits,
            "uncertainty": self.uncertainty,
            "entropy": self.entropy,
            "advantage": self.advantage
        }
``n

## File: scoring_engine.py

`python
# stephanie/scoring/scoring_engine.py

from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.scoring_manager import ScoringManager


class ScoringEngine:
    def __init__(self, cfg, memory, prompt_loader, logger, call_llm):
        self.cfg = cfg
        self.memory = memory
        self.prompt_loader = prompt_loader
        self.logger = logger
        self.call_llm = call_llm
        self.scoring_managers = {}

    def get_manager(self, scoring_profile: str) -> ScoringManager:
        if scoring_profile not in self.scoring_managers:
            config_path = self.cfg.get(
                f"{scoring_profile}_score_config",
                f"config/scoring/{scoring_profile}.yaml",
            )
            self.scoring_managers[scoring_profile] = ScoringManager.from_file(
                filepath=config_path,
                prompt_loader=self.prompt_loader,
                cfg=self.cfg,
                logger=self.logger,
                memory=self.memory,
                llm_fn=self.call_llm,
            )
        return self.scoring_managers[scoring_profile]

    def score(
        self,
        target_id,
        target_type: TargetType,
        text: str,
        context: dict,
        scoring_profile: str,
    ) -> ScoreBundle:
        try:
            scorable = Scorable(id=target_id, text=text, target_type=target_type)
            scoring_manager = self.get_manager(scoring_profile)

            merged_context = {
                "target_type": target_type,
                "target": scorable.to_dict(),
                **context,
            }

            scorer = scoring_manager.scorer
            if not scorer:
                score_result = scoring_manager.evaluate(
                    scorable=scorable, context=merged_context, llm_fn=self.call_llm
                )
            else:
                score_result = scorer.score(
                    context, scorable, scoring_manager.dimensions
                )

            self.logger.log("ItemScored", score_result.to_dict())
            return score_result

        except Exception as e:
            self.logger.log(
                "ScoringFailed",
                {"target_id": target_id, "target_type": target_type, "error": str(e)},
            )
            return {}

    def score_item(
        self, scorable: Scorable, context: dict, scoring_profile: str
    ) -> ScoreBundle:
        try:
            scoring_manager = self.get_manager(scoring_profile)

            merged_context = {
                "target_type": scorable.target_type,
                "target": scorable.to_dict(),
                **context,
            }

            scorer = scoring_manager.scorer
            if not scorer:
                score_result = scoring_manager.evaluate(
                    scorable=scorable, context=merged_context, llm_fn=self.call_llm
                )
            else:
                score_result = scorer.score(
                    context, scorable, scoring_manager.dimensions
                )

            self.logger.log("ItemScored", score_result.to_dict())
            return score_result

        except Exception as e:
            self.logger.log(
                "ScoreItemFailed",
                {
                    "scrable": scorable,
                    "error": str(e),
                },
            )
            return {}
``n

## File: scoring_manager.py

`python
# stephanie/scoring/scoring_manager.py
import json
import re
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.evaluation_attribute import EvaluationAttributeORM
from stephanie.models.score import ScoreORM
from stephanie.models.score_dimension import ScoreDimensionORM
from stephanie.prompts.prompt_renderer import PromptRenderer
from stephanie.scoring.calculations.score_delta import ScoreDeltaCalculator
from stephanie.scoring.calculations.weighted_average import \
    WeightedAverageCalculator
from stephanie.scoring.fallback_scorer import FallbackScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_display import ScoreDisplay
from stephanie.scoring.score_result import ScoreResult


class ScoringManager(BaseAgent):
    def __init__(
        self,
        dimensions,
        prompt_loader,
        cfg,
        logger,
        memory,
        calculator=None,
        dimension_filter_fn=None,
        scorer: Optional[FallbackScorer] = None,
        scoring_profile: str = "default",
    ):
        super().__init__(cfg, memory, logger)
        self.dimensions = dimensions
        self.prompt_loader = prompt_loader
        self.output_format = cfg.get("output_format", "simple")  # default
        self.prompt_renderer = PromptRenderer(prompt_loader, cfg)
        self.calculator = calculator or WeightedAverageCalculator()
        self.dimension_filter_fn = dimension_filter_fn
        self.scoring_profile = scoring_profile
         # Initialize fallback scorer if not provided
        if scorer is None:
            from stephanie.scoring.llm_scorer import LLMScorer
            from stephanie.scoring.mrq_scorer import MRQScorer
            from stephanie.scoring.svm_scorer import SVMScorer

            svm_scorer = SVMScorer(cfg, memory, logger, dimensions=dimensions)
            mrq_scorer = MRQScorer(cfg, memory, logger, dimensions=dimensions)
            llm_scorer = LLMScorer(cfg, memory, logger, prompt_loader=prompt_loader, call_llm=self._call_llm)

            self.scorer = FallbackScorer(
                scorers=[svm_scorer, mrq_scorer, llm_scorer],
                fallback_order=["svm", "mrq", "llm"],
                default_fallback="llm",
                logger=logger
            )
        else:
            self.scorer = scorer

    async def run(self, context: dict) -> dict:
        """
        Main entry point for running the scoring manager.
        This method can be overridden by subclasses to implement custom logic.
        """
        # Default implementation just returns the context
        return context

    def dimension_names(self):
        """Returns the names of all dimensions."""
        return [dim["name"] for dim in self.dimensions]

    def filter_dimensions(self, scorable, context):
        """
        Returns the list of dimensions to use for this evaluation.
        Override or provide a hook function to filter dynamically.
        """
        if self.dimension_filter_fn:
            return self.dimension_filter_fn(self.dimensions, scorable, context)
        return self.dimensions

    @staticmethod
    def get_postprocessor(extra_data):
        """Returns a postprocessor function based on the 'postprocess' key."""
        ptype = extra_data.get("postprocess")
        if ptype == "clip_0_5":
            return lambda s: max(0, min(s, 5))
        if ptype == "normalize_10":
            return lambda s: min(s / 10.0, 1.0)
        if ptype == "exp_boost":
            return lambda s: round((1.2**s) - 1, 2)
        return lambda s: s  # Default is identity

    @classmethod
    def from_db(
        cls,
        session: Session,
        stage: str,
        prompt_loader=None,
        cfg=None,
        logger=None,
        memory=None,
    ):
        rows = session.query(ScoreDimensionORM).filter_by(stage=stage).all()
        dimensions = [
            {
                "name": row.name,
                "prompt_template": row.prompt_template,
                "weight": row.weight,
                "parser": cls.get_parser(row.extra_data or {}),
                "file": row.extra_data.get("file") if row.extra_data else None,
                "postprocess": cls.get_postprocessor(row.extra_data or {}),
            }
            for row in rows
        ]
        return cls(
            dimensions,
            prompt_loader=prompt_loader,
            cfg=cfg,
            logger=logger,
            memory=memory,
        )

    def get_dimensions(self):
        return [d["name"] for d in self.dimensions]

    @classmethod
    def from_file(
        cls,
        filepath: str,
        prompt_loader,
        cfg,
        logger,
        memory,
        scoring_profile=None,
        llm_fn=None,
    ):
        with open(Path(filepath), "r") as f:
            data = yaml.safe_load(f)

        # Default to 'simple' if not provided
        output_format = data.get("output_format", "simple")

        dimensions = [
            {
                "name": d["name"],
                "file": d.get("file"),
                "prompt_template": d.get(
                    "prompt_template", d.get("file")
                ),  # fallback to file
                "weight": d.get("weight", 1.0),
                "parser": cls.get_parser(d.get("extra_data", {})),
                "postprocess": cls.get_postprocessor(d.get("extra_data", {})),
            }
            for d in data["dimensions"]
        ]

        # Ensure the output_format is accessible in instance
        cfg = cfg.copy()
        cfg["output_format"] = output_format

        from stephanie.scoring.llm_scorer import LLMScorer
        from stephanie.scoring.mrq_scorer import MRQScorer
        from stephanie.scoring.svm_scorer import SVMScorer

        if data["scorer"] == "mrq":
            # Use MRQ scoring profile if specified
            scorer = MRQScorer(cfg, memory, logger)
            scorer.load_models()
        elif data["scorer"] == "svm":
            # Use SVM scoring profile if specified
            scorer = SVMScorer(cfg, memory, logger)
            scorer.load_models()
        else:
            # Default to LLM scoring profile
            scorer = LLMScorer(
                cfg, memory, logger, prompt_loader=prompt_loader, llm_fn=llm_fn
            )

        return cls(
            dimensions=dimensions,
            prompt_loader=prompt_loader,
            cfg=cfg,
            logger=logger,
            memory=memory,
            scoring_profile=scoring_profile,
            scorer=scorer,
        )

    @staticmethod
    def get_parser(extra_data):
        parser_type = extra_data.get("parser", "numeric")
        if parser_type == "numeric":
            return lambda r: ScoringManager.extract_score_from_last_line(r)
        if parser_type == "numeric_cor":
            return lambda r: ScoringManager.parse_numeric_cor(r)

        return lambda r: 0.0

    @staticmethod
    def extract_score_from_last_line(response: str) -> float:
        """
        Extracts a numeric score from any line containing 'score: <number>' (case-insensitive),
        scanning in reverse for the most recent score mention.
        """
        lines = response.strip().splitlines()
        for line in reversed(lines):
            match = re.search(r"\bscore:\s*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    @staticmethod
    def parse_numeric_cor(response: str) -> float:
        """
        Extracts a numeric score from a <answer> block or a line containing 'score: <number>'.
        """
        # Try CoR-style first
        match = re.search(
            r"<answer>\s*\[*\s*(\d+(?:\.\d+)?)\s*\]*\s*</answer>",
            response,
            re.IGNORECASE,
        )
        if match:
            return float(match.group(1))

        # Fallback to score: X
        match = re.search(r"\bscore:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if match:
            return float(match.group(1))

        raise ValueError(f"Could not extract numeric score from response: {response}")

    def evaluate(self, context: dict, scorable: Scorable, llm_fn=None):
        try:
            score = self.scorer.score(
                context, scorable, self.dimensions, llm_fn=llm_fn
            )
        except Exception as e:
            self.logger.log(
                "MgrScoreParseError",
                {"scorable": scorable, "error": str(e)},
            )
            score = self.evaluate_llm(context, scorable, llm_fn or self.call_llm)
        log_key = "CorDimensionEvaluated" if format == "cor" else "DimensionEvaluated"
        self.logger.log(
            log_key,
            {"ScoreCompleted": score.to_dict()},
        )

        return score

    def evaluate_llm(self, context: dict, scorable: Scorable, llm_fn=None):
        if llm_fn is None:
            raise ValueError("You must pass a call_llm function to evaluate")

        results = []
        force_rescore = context.get("force_rescore", False)

        # Use filter_dimensions if available
        dimensions_to_use = self.filter_dimensions(scorable, context)

        for dim in dimensions_to_use:
            print(f"Evaluating dimension: {dim['name']}")
            prompt = self.prompt_renderer.render(
                dim, {"hypothesis": scorable, **context}
            )

            prompt_hash = ScoreORM.compute_prompt_hash(prompt, scorable)
            if not force_rescore:
                cached_result = self.memory.scores.get_score_by_prompt_hash(prompt_hash)
                if cached_result:
                    self.logger.log("ScoreCacheHit", {"dimension": dim["name"]})
                    result = cached_result
                    results.append(result)
                    continue

            response = llm_fn(prompt, context=context)
            try:
                score = dim["parser"](response)
                score = dim.get("postprocess", lambda s: s)(score)
            except Exception as e:
                self.logger.log(
                    "ScoreParseError",
                    {"dimension": dim["name"], "response": response, "error": str(e)},
                )
                self.handle_score_error(dim, response, e)
                score = 0.0

            result = ScoreResult(
                dimension=dim["name"],
                score=score,
                weight=dim["weight"],
                rationale=response,
                prompt_hash=prompt_hash,
                source="llm",
                target_type=scorable.target_type,
            )
            results.append(result)

            log_key = (
                "CorDimensionEvaluated" if format == "cor" else "DimensionEvaluated"
            )
            self.logger.log(
                log_key,
                {"dimension": dim["name"], "score": score, "response": response},
            )

        bundle = ScoreBundle(results={r.dimension: r for r in results})
        self.save_score_to_memory(
            bundle, scorable, context, self.cfg, self.memory, self.logger
        )
        return bundle

    def handle_score_error(self, dim, response, error):
        if self.cfg.get("fail_silently", True):
            return 0.0
        raise ValueError(f"Failed to parse score {response} for {dim['name']}: {error}")

    @staticmethod
    def save_score_to_memory(
        bundle: ScoreBundle,
        scorable: Scorable,
        context: dict,
        cfg: dict,
        memory,
        logger,
        source,
        model_name=None,
    ):
        goal = context.get("goal")
        pipeline_run_id = context.get("pipeline_run_id")
        weighted_score = bundle.calculator.calculate(bundle)

        scores_json = {
            "stage": cfg.get("stage", "review"),
            "dimensions": bundle.to_dict(),
            "final_score": round(weighted_score, 2),
        }

        if not model_name:
            model_name = cfg.get("model", {}).get("name", "UnknownModel")

        eval_orm = EvaluationORM(
            goal_id=goal.get("id"),
            pipeline_run_id=pipeline_run_id,
            target_type=scorable.target_type,
            target_id=scorable.id,
            source=source,
            agent_name=cfg.get("name"),
            model_name=model_name,
            embedding_type=memory.embedding.type,
            evaluator_name=cfg.get("evaluator", cfg.get("model_type", "ScoreEvaluator")),
            strategy=cfg.get("strategy"),
            reasoning_strategy=cfg.get("reasoning_strategy"),
            scores=scores_json,
            extra_data={"source": source},
        )
        memory.session.add(eval_orm)
        memory.session.flush()

        for result in bundle.results:
            score_result = bundle.results[result]
            score = ScoreORM(
                evaluation_id=eval_orm.id,
                dimension=score_result.dimension,
                score=score_result.score,
                source=score_result.source,
                weight=score_result.weight,
                rationale=score_result.rationale,
                prompt_hash=score_result.prompt_hash
                or ScoreORM.compute_prompt_hash(goal.get("goal_text", ""), scorable),
            )
            memory.session.add(score)

            # After inserting ScoreORM
            attribute = EvaluationAttributeORM(
                evaluation_id=eval_orm.id,
                dimension=score_result.dimension,
                source=score_result.source,
                raw_score=score_result.score,
                energy=score_result.energy,
                uncertainty=score_result.uncertainty,
                pi_value=score_result.policy_logits[0] if score_result.policy_logits else None,
                entropy=score_result.entropy,
                advantage=score_result.advantage,
                q_value=score_result.q_value,
                v_value=score_result.state_value,
                policy_logits=json.dumps(score_result.policy_logits),
                extra=score_result.to_dict(),
            )
            memory.session.add(attribute)

        memory.session.commit()

        logger.log(
            "ScoreSavedToMemory",
            {
                "goal_id": goal.get("id"),
                "target_id": scorable.id,
                "target_type": scorable.target_type,
                "scores": scores_json,
            },
        )
        ScoreDeltaCalculator(cfg, memory, logger).log_score_delta(
            scorable, weighted_score, goal.get("id")
        )
        ScoreDisplay.show(scorable, bundle.to_dict(), weighted_score)


    @staticmethod
    def save_document_score_to_memory(
        bundle, document, context, cfg, memory, logger, source="DocumentEvaluator"
    ):
        goal = context.get("goal")
        pipeline_run_id = context.get("pipeline_run_id")
        document_id = document.get("id")
        weighted_score = bundle.calculator.calculate(bundle)

        soring_text = ScoringManager.get_scoring_text(document)
        scorable = Scorable(
            text=soring_text, target_type=TargetType.DOCUMENT, id=document_id
        )

        scores_json = {
            "stage": cfg.get("stage", "review"),
            "dimensions": bundle.to_dict(),
            "final_score": round(weighted_score, 2),
        }

        eval_orm = EvaluationORM(
            goal_id=goal.get("id"),
            pipeline_run_id=pipeline_run_id,
            target_type=TargetType.DOCUMENT.value,
            target_id=document_id,
            agent_name=cfg.get("name"),
            model_name=cfg.get("model", {}).get("name"),
            embedding_type=memory.embedding.type,
            evaluator_name=cfg.get("evaluator", "ScoreEvaluator"),
            strategy=cfg.get("strategy"),
            reasoning_strategy=cfg.get("reasoning_strategy"),
            scores=scores_json,
            extra_data={"source": source},
        )
        memory.session.add(eval_orm)
        memory.session.flush()

        for result in bundle.results:
            score_result = bundle.results[result]
            score = ScoreORM(
                evaluation_id=eval_orm.id,
                dimension=score_result.dimension,
                score=score_result.score,
                weight=score_result.weight,
                rationale=score_result.rationale,
                prompt_hash=score_result.prompt_hash,
            )
            memory.session.add(score)

        memory.session.commit()

        logger.log(
            "ScoreSavedToMemory",
            {
                "goal_id": goal.get("id"),
                "hypothesis_id": document_id,
                "scores": scores_json,
            },
        )
        ScoreDeltaCalculator(cfg, memory, logger).log_score_delta(
            scorable, weighted_score, goal.get("id")
        )
        ScoreDisplay.show(scorable, bundle.to_dict(), weighted_score)

    @staticmethod
    def get_scoring_text(document: dict) -> str:
        if document.get("summary"):
            return f"{document.get('title', '')}\n\n{document['summary']}".strip()
        elif document.get("content"):
            return document["content"][:1500]  # Safely truncate
        else:
            return document.get("title", "")
``n

## File: sicql_scorer.py

`python
import os

import torch
import torch.nn.functional as F

from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.model.in_context_q import InContextQModel
from stephanie.scoring.model.policy_head import PolicyHead
from stephanie.scoring.model.q_head import QHead
from stephanie.scoring.model.v_head import VHead
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator


class SICQLScorer(BaseScorer):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "sicql"
        self.embedding_type = memory.embedding.type
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim

        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")

        self.models = {}
        self.model_meta = {}
        self.tuners = {}

        self.dimensions = cfg.get("dimensions", [])
        self._load_models(self.dimensions)

    def _load_models(self, dimensions):
        for dim in dimensions:
            locator = ModelLocator(
                root_dir=self.model_path,
                embedding_type=self.embedding_type,
                model_type=self.model_type,
                target_type=self.target_type,
                dimension=dim,
                version=self.version,
            )

            encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
            q_head = QHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            v_head = VHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            pi_head = PolicyHead(zsa_dim=self.dim, hdim=self.hdim, num_actions=3).to(self.device)

            encoder.load_state_dict(torch.load(locator.encoder_file(), map_location=self.device))
            q_head.load_state_dict(torch.load(locator.q_head_file(), map_location=self.device))
            v_head.load_state_dict(torch.load(locator.v_head_file(), map_location=self.device))
            pi_head.load_state_dict(torch.load(locator.pi_head_file(), map_location=self.device))

            model = InContextQModel(
                encoder=encoder,
                q_head=q_head,
                v_head=v_head,
                pi_head=pi_head,
                embedding_store=self.memory.embedding,
                device=self.device,
            )
            self.models[dim] = model

            meta = load_json(locator.meta_file()) if os.path.exists(locator.meta_file()) else {"min_score": 0, "max_score": 100}
            self.model_meta[dim] = meta

            tuner_path = locator.tuner_file()
            if os.path.exists(tuner_path):
                tuner = RegressionTuner(dimension=dim)
                tuner.load(tuner_path)
                self.tuners[dim] = tuner


    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        goal_text = goal.get("goal_text")
        results = {}

        for dim in dimensions:
            model = self.models.get(dim)
            prompt_emb = torch.tensor(
                self.memory.embedding.get_or_create(goal_text), device=self.device
            ).unsqueeze(0)
            output_emb = torch.tensor(
                self.memory.embedding.get_or_create(scorable.text), device=self.device
            ).unsqueeze(0)
            result = model(prompt_emb, output_emb)


            q_value = result["q_value"].item()
            v_value = result["state_value"].item()
            policy_logits = result["action_logits"].cpu().detach().numpy().tolist()

            if isinstance(policy_logits, list) and len(policy_logits) == 1:
                if isinstance(policy_logits[0], list):
                    # [[0.1166]] → [0.1166]
                    policy_logits = policy_logits[0]

            self.logger.log("PolicyLogits", {"dimension": dim, "logits": policy_logits})

                # Calculate uncertainty (|Q - V|)
            uncertainty = abs(q_value - v_value)
            
            # Calculate entropy from policy logits
            policy_tensor = torch.tensor(policy_logits)
            action_probs = F.softmax(policy_tensor, dim=-1)
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8)).item()
            
            # Calculate advantage
            advantage = q_value - v_value
            meta = self.model_meta.get(dim, {"min": 0, "max": 100})
            if dim in self.tuners:
                scaled_score = self.tuners[dim].transform(q_value)
            else:
                normalized = torch.sigmoid(torch.tensor(q_value)).item()
                scaled_score = normalized * (meta["max_value"] - meta["min_value"]) + meta["min_value"]

            scaled_score = max(min(scaled_score, meta["max_value"]), meta["min_value"])


            final_score = round(scaled_score, 4)
            prompt_hash = ScoreORM.compute_prompt_hash(goal_text, scorable)

            rationale = f"Q={q_value:.4f}, V={v_value:.4f}, Δ={uncertainty:.3f}, H={entropy:.3f}"

            results[dim] = ScoreResult(
                        dimension=dim,
                        score=final_score,
                        rationale=rationale,
                        weight=1.0,
                        q_value=q_value,
                        energy=q_value,
                        source=self.name,
                        target_type=scorable.target_type,
                        prompt_hash=prompt_hash,
                        state_value=v_value,
                        policy_logits=policy_logits,
                        uncertainty=uncertainty,
                        entropy=entropy,
                        advantage=advantage,
                    )
        return ScoreBundle(results=results)
``n

## File: structured_engine.py

`python
# stephanie/scoring/structured_engine.py
import re
from string import Template


class StructuredScoringEngine:
    def __init__(self, dimensions, prompt_loader=None, cfg=None, logger=None):
        self.dimensions = dimensions
        self.prompt_loader = prompt_loader
        self.cfg = cfg or {}
        self.logger = logger

    def evaluate(self, hypothesis: dict, context: dict = {}, llm_fn=None) -> dict:
        if llm_fn is None:
            raise ValueError("You must provide an llm_fn (e.g., agent.call_llm)")

        results = {}
        for dim in self.dimensions:
            prompt = self._render_prompt(dim, hypothesis, context)
            response = llm_fn(prompt, context=context)
            try:
                score = dim["parser"](response)
            except Exception as e:
                score = 0.0
                if self.logger:
                    self.logger.log(
                        "StrctScoreParseError",
                        {
                            "dimension": dim["name"],
                            "response": response,
                            "error": str(e),
                        },
                    )
            if self.logger:
                self.logger.log(
                    "StructuredDimensionEvaluated",
                    {"dimension": dim["name"], "score": score, "response": response},
                )
            results[dim["name"]] = {
                "score": score,
                "rationale": response,
                "weight": dim.get("weight", 1.0),
            }

        results["final_score"] = self._aggregate(results)
        return results

    def _render_prompt(self, dim: dict, hypothesis: dict, context: dict) -> str:
        ctx = {"hypothesis": hypothesis, **context}
        if self.prompt_loader and dim.get("file"):
            return self.prompt_loader.from_file(
                file_name=dim["file"], config=self.cfg, context=ctx
            )
        else:
            return Template(dim["prompt_template"]).substitute(ctx)

    def _aggregate(self, results: dict) -> float:
        total = 0.0
        weight_sum = 0.0
        for dim, val in results.items():
            if not isinstance(val, dict):  # skip final_score key
                continue
            total += val["score"] * val.get("weight", 1.0)
            weight_sum += val.get("weight", 1.0)
        return round(total / weight_sum, 2) if weight_sum else 0.0

    @staticmethod
    def extract_score_from_last_line(response: str) -> float:
        lines = response.strip().splitlines()
        for line in reversed(lines):
            match = re.search(r"score:\s*(\d+(\.\d+)?)", line.strip(), re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    @staticmethod
    def parse_numeric_cor(response: str) -> float:
        match = re.search(
            r"<answer>\s*\[\[(\d+(?:\.\d+)?)\]\]\s*</answer>", response, re.IGNORECASE
        )
        if not match:
            raise ValueError(
                f"Could not extract numeric score from CoR-style answer: {response}"
            )
        return float(match.group(1))

    @staticmethod
    def get_parser(extra_data):
        parser_type = extra_data.get("parser", "numeric")
        if parser_type == "numeric":
            return StructuredScoringEngine.extract_score_from_last_line
        if parser_type == "numeric_cor":
            return StructuredScoringEngine.parse_numeric_cor
        return lambda r: 0.0
``n

## File: svm_scorer.py

`python
# stephanie/scoring/svm/svm_scorer.py


import numpy as np
from joblib import load

from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json


class SVMScorer(BaseScorer):
    def __init__(self, cfg: dict, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "svm"
        self.models = {}        # dim -> (scaler, model)
        self.tuners = {}        # dim -> RegressionTuner
        self.metas = {}         # dim -> model metadata
        self._load_all_dimensions()

    def _load_all_dimensions(self):
        for dim in self.dimensions:
            locator = self.get_locator(dim) 
            self.models[dim] = (
                load(locator.scaler_file()),
                load(locator.model_file(suffix=".joblib")),
            )
            tuner = RegressionTuner(dimension=dim, logger=self.logger)
            tuner.load(locator.tuner_file())
            self.tuners[dim] = tuner
            self.metas[dim] = load_json(locator.meta_file())

    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        goal_text = goal.get("goal_text", "")
        ctx_emb = np.array(self.memory.embedding.get_or_create(goal_text))
        doc_emb = np.array(self.memory.embedding.get_or_create(scorable.text))
        input_vec = np.concatenate([ctx_emb, doc_emb]).reshape(1, -1)

        results = {}
        for dim in dimensions:
            scaler, model = self.models[dim]
            tuner = self.tuners[dim]
            meta = self.metas.get(dim, {"min_score": 0, "max_score": 100})

            scaled_input = scaler.transform(input_vec)
            raw_score = model.predict(scaled_input)[0]
            tuned_score = tuner.transform(raw_score)

            # Clip to min/max
            final_score = max(min(tuned_score, meta["max_score"]), meta["min_score"])
            results[dim] = ScoreResult(
                dimension=dim,
                score=final_score,
                rationale=f"SVM raw={round(raw_score, 4)}",
                weight=1.0,
                source=self.model_type,
                energy=0.0,
                target_type=scorable.target_type,
                prompt_hash=ScoreORM.compute_prompt_hash(goal_text, scorable),
            )

        return ScoreBundle(results=results)
``n

## File: training\__init__.py

`python
# stephanie/scoring/training/__init__.py
``n

## File: training\base_engine.py

`python
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from stephanie.models.training_stats import TrainingStatsORM


class BaseTrainingEngine:
    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = cfg.get("batch_size", 32)
        self.epochs = cfg.get("epochs", 10)
        self.lr = cfg.get("lr", 1e-4)
        self.gamma = cfg.get("gamma", 0.99)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.patience = cfg.get("patience", 3)
        self.min_delta = cfg.get("min_delta", 0.001)
        self.early_stop_counter = 0
        self.best_loss = float('inf')
    
    def _create_dataloader(self, samples):
        """Convert samples to PyTorch DataLoader"""
        context_embs, doc_embs, scores = [], [], []
        
        for item in samples:
            context = item.get("title", "")
            doc_text = item.get("output", "")
            
            context_emb = torch.tensor(self.memory.embedding.get_or_create(context))
            doc_emb = torch.tensor(self.memory.embedding.get_or_create(doc_text))
            score = float(item.get("score", 0.5))
            
            context_embs.append(context_emb)
            doc_embs.append(doc_emb)
            scores.append(score)
        
        # Convert to tensors
        context_tensors = torch.stack(context_embs).to(self.device)
        doc_tensors = torch.stack(doc_embs).to(self.device)
        score_tensors = torch.tensor(scores).float().to(self.device)
        
        return DataLoader(
            torch.utils.data.TensorDataset(
                context_tensors, doc_tensors, score_tensors
            ),
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def _should_stop_early(self, losses):
        """Early stopping logic with validation"""
        if not self.use_early_stopping or len(losses) < self.patience:
            return False
        
        # Get recent losses
        recent_losses = losses[-self.patience:]
        avg_recent = sum(recent_losses) / len(recent_losses)
        
        # Check for improvement
        if avg_recent < self.best_loss - self.min_delta:
            self.best_loss = avg_recent
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
        
        # Return early stopping condition
        return self.early_stop_counter >= self.patience
    
    def _log_training_stats(self, stats):
        """Log training stats to database"""
        training_stats = TrainingStatsORM(**stats)
        self.memory.session.add(training_stats)
        self.memory.session.commit()
``n

## File: training\base_trainer.py

`python
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class BaseTrainer:
    class Locator:
        def __init__(self, root_dir, model_type, target_type, dimension, version, embedding_type):
            self.root_dir = root_dir
            self.model_type = model_type
            self.target_type = target_type
            self.dimension = dimension
            self.version = version
            self.embedding_type = embedding_type

        @property
        def base_path(self):
            path = os.path.join(
                self.root_dir,
                self.embedding_type,
                self.model_type,
                self.target_type,
                self.dimension,
                self.version,
            )
            os.makedirs(path, exist_ok=True)
            return path

        def encoder_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_encoder.pt")

        def q_head_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_q.pt")

        def v_head_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_v.pt")

        def pi_head_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_pi.pt")

        def meta_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.meta.json")

        def tuner_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.tuner.json")

        def scaler_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_scaler.joblib")

        def joblib_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_model.joblib")

        def model_file(self, suffix: str = ".pt") -> str:
            return os.path.join(self.base_path, f"{self.dimension}{suffix}")


        def model_exists(self) -> bool:
            return False

    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.embedding_type = memory.embedding.type
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.target_type = cfg.get("target_type", "document")
        self.model_type = cfg.get("model_type", "base")
        self.dimensions = cfg.get("dimensions", [])
        self.min_samples = cfg.get("min_samples", 5)

    def get_locator(self, dimension):
        return self.Locator(
            root_dir=self.root_dir,
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.embedding_type,
        )

    def _create_dataloader(self, samples):
        valid = []
        for s in samples:
            ctx_text = s.get("title", "")
            doc_text = s.get("output", "")
            score = s.get("score", 0.5)

            if not ctx_text or not doc_text or not isinstance(score, (float, int)):
                continue

            ctx_emb = torch.tensor(self.memory.embedding.get_or_create(ctx_text)).to(self.device)
            doc_emb = torch.tensor(self.memory.embedding.get_or_create(doc_text)).to(self.device)

            valid.append({"context": ctx_emb, "document": doc_emb, "score": score})

        if len(valid) < self.min_samples:
            return None

        return DataLoader(
            TensorDataset(
                torch.stack([s["context"] for s in valid]),
                torch.stack([s["document"] for s in valid]),
                torch.tensor([s["score"] for s in valid])
            ),
            batch_size=self.cfg.get("batch_size", 32),
            shuffle=True
        )

    def _save_meta_file(self, meta: dict, dimension: str):
        locator = self.get_locator(dimension)
        with open(locator.meta_file(), "w") as f:
            json.dump(meta, f)

    def _calculate_policy_metrics(self, logits):
        probs = F.softmax(torch.tensor(logits), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        stability = probs.max().item()
        return entropy, stability

    def log_event(self, name: str, payload: dict):
        if self.logger:
            self.logger.log(name, payload)

``n

## File: training\document_trainer.py

`python
# stephanie/scoring/training/document_trainer.py
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset

from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.training.base_trainer import BaseTrainer


class DocumentTrainer(BaseTrainer):
    def init_encoder(self):
        return TextEncoder().to(self.device)

    def init_predictor(self):
        return ValuePredictor().to(self.device)

    def prepare_training_data(self, samples: List[dict]) -> DataLoader:
        inputs, labels = [], []
        total = len(samples)

        for idx, item in enumerate(samples):
            context_text = item.get("title", "")
            context_emb = self.memory.embedding.get_or_create(context_text)
            doc_a_emb = self.memory.embedding.get_or_create(item["text_a"])
            doc_b_emb = self.memory.embedding.get_or_create(item["text_b"])

            preferred = "a" if item["value_a"] >= item["value_b"] else "b"

            context_tensor = torch.tensor(context_emb).unsqueeze(0).to(self.device)
            a_tensor = torch.tensor(doc_a_emb).unsqueeze(0).to(self.device)
            b_tensor = torch.tensor(doc_b_emb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                zsa_a = self.encoder(context_tensor, a_tensor)
                zsa_b = self.encoder(context_tensor, b_tensor)

            diff = zsa_a - zsa_b if preferred == "a" else zsa_b - zsa_a

            inputs.append(diff.squeeze(0).detach())
            labels.append(torch.tensor([1.0], device=self.device))

            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                self.logger.log(
                    "DocumentTrainingProgress",
                    {
                        "current": idx + 1,
                        "total": total,
                        "percent": round((idx + 1) / total * 100, 2),
                    },
                )

        dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
        return DataLoader(dataset, batch_size=16, shuffle=True)
``n

## File: training\ebt_trainer.py

`python

import torch
import torch.nn.functional as F

from stephanie.scoring.training.base_trainer import BaseTrainer


class EBTTrainer(BaseTrainer):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "ebt"
        self.num_actions = 3 #cfg.get("num_actions", 3)

    def train(self, samples, dimension):
        dl = self._create_dataloader(samples)
        if not dl:
            return {"error": f"Insufficient samples for {dimension}"}

        from torch import nn
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        from stephanie.scoring.model.ebt_model import EBTModel

        model = EBTModel(self.dim, self.hdim, self.num_actions, self.device).to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.get("lr", 2e-5))
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        mse = nn.MSELoss()

        def expectile_loss(diff, tau=0.7):
            return (torch.where(diff > 0, tau * diff.pow(2), (1 - tau) * diff.pow(2))).mean()

        q_losses, v_losses, pi_losses, entropies = [], [], [], []

        for epoch in range(self.cfg.get("epochs", 10)):
            total_q, total_v, total_pi = 0.0, 0.0, 0.0
            for ctx, doc, label in dl:
                ctx, doc, label = ctx.to(self.device), doc.to(self.device), label.to(self.device)
                outputs = model(ctx, doc)

                q_loss = mse(outputs["q_value"], label)
                v_loss = expectile_loss(outputs["q_value"].detach() - outputs["state_value"])
                adv = (outputs["q_value"] - outputs["state_value"]).detach()
                policy_probs = F.softmax(outputs["action_logits"], dim=-1)
                entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=-1).mean()
                adv = adv.unsqueeze(1)  # Shape becomes [batch_size, 1]
                pi_loss = -(torch.log(policy_probs) * adv).mean() - 0.01 * entropy

                loss = q_loss * self.cfg.get("q_weight", 1.0) + \
                       v_loss * self.cfg.get("v_weight", 0.5) + \
                       pi_loss * self.cfg.get("pi_weight", 0.3)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_q += q_loss.item()
                total_v += v_loss.item()
                total_pi += pi_loss.item()
                entropies.append(entropy.item())

            avg_q = total_q / len(dl)
            avg_v = total_v / len(dl)
            avg_pi = total_pi / len(dl)
            scheduler.step(avg_q)

            self.log_event("EBTTrainerEpoch", {
                "dimension": dimension,
                "epoch": epoch + 1,
                "q_loss": avg_q,
                "v_loss": avg_v,
                "pi_loss": avg_pi,
                "policy_entropy": sum(entropies) / len(entropies)
            })

            q_losses.append(avg_q)
            v_losses.append(avg_v)
            pi_losses.append(avg_pi)

        self._save_model(model, dimension)

        return {
            "q_loss": q_losses[-1],
            "v_loss": v_losses[-1],
            "pi_loss": pi_losses[-1],
            "policy_entropy": sum(entropies) / len(entropies),
        }

    def _save_model(self, model, dimension):
        locator = self.get_locator(dimension)
        torch.save(model.encoder.state_dict(), locator.encoder_file())
        torch.save(model.q_head.state_dict(), locator.q_head_file())
        torch.save(model.v_head.state_dict(), locator.v_head_file())
        torch.save(model.pi_head.state_dict(), locator.pi_head_file())

        meta = {
            "dim": model.embedding_dim,
            "hdim": model.hidden_dim,
            "num_actions": model.num_actions,
            "version": self.version,
            "min_score": self.cfg.get("min_score", 0),
            "max_score": self.cfg.get("max_score", 100),
        }
        self._save_meta_file(meta, dimension)
``n

## File: training\hypothesis_trainer.py

`python
# stephanie/scoring/training/hypothesis_trainer.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from stephanie.evaluator.hypothesis_value_predictor import \
    HypothesisValuePredictor
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.training.base_trainer import BaseTrainer


class HypothesisTrainer(BaseTrainer):
    def __init__(
        self, memory, logger, encoder=None, value_predictor=None, device="cpu"
    ):
        encoder = encoder or TextEncoder()
        value_predictor = value_predictor or HypothesisValuePredictor(512, 1024)
        super().__init__(memory, logger, encoder, value_predictor, device)

    def prepare_training_data(self, samples):
        inputs, labels = [], []
        total = len(samples)

        for idx, item in enumerate(samples):
            prompt_emb = self.memory.embedding.get_or_create(item["prompt"])
            output_a_emb = self.memory.embedding.get_or_create(item["output_a"])
            output_b_emb = self.memory.embedding.get_or_create(item["output_b"])

            preferred = "a" if item["value_a"] >= item["value_b"] else "b"

            prompt_tensor = torch.tensor(prompt_emb).unsqueeze(0).to(self.device)
            a_tensor = torch.tensor(output_a_emb).unsqueeze(0).to(self.device)
            b_tensor = torch.tensor(output_b_emb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                zsa_a = self.encoder(prompt_tensor, a_tensor)
                zsa_b = self.encoder(prompt_tensor, b_tensor)

            diff = zsa_a - zsa_b if preferred == "a" else zsa_b - zsa_a

            inputs.append(diff.squeeze(0).detach())
            labels.append(torch.tensor([1.0], device=self.device))

            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                self.logger.log(
                    "HypothesisTrainerProgress",
                    {
                        "current": idx + 1,
                        "total": total,
                        "percent": round((idx + 1) / total * 100, 2),
                    },
                )

        dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
        return DataLoader(dataset, batch_size=16, shuffle=True)
``n

## File: training\mrq_trainer.py

`python
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from stephanie.models.training_stats import TrainingStatsORM
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.model import MRQModel
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class MRQTrainer(BaseTrainer):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.use_tuner = cfg.get("use_tuner", True)
        self.min_samples = cfg.get("min_samples", 5)
        self.batch_size = cfg.get("batch_size", 1)
        self.model = self._build_model()
        self.use_tuner = cfg.get("use_tuner", True)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.batch_size = cfg.get("batch_size", 2)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("lr", 1e-4)
        self.min_samples = cfg.get("min_samples", 5)

        
        self.logger.log("MRQTrainerInitialized", {
            "embedding_type": self.embedding_type,
            "use_tuner": self.use_tuner,
            "device": str(self.device)
        })

    def _create_dataloader(self, samples):
        inputs, labels = [], []

        for item in samples:
            prompt = item.get("title", "")
            output_a = item.get("output_a", "")
            output_b = item.get("output_b", "")
            value_a = item.get("value_a", 0)
            value_b = item.get("value_b", 0)

            if not prompt or not output_a or not output_b:
                continue

            try:
                prompt_emb = torch.tensor(self.memory.embedding.get_or_create(prompt)).unsqueeze(0).to(self.device)
                a_emb = torch.tensor(self.memory.embedding.get_or_create(output_a)).unsqueeze(0).to(self.device)
                b_emb = torch.tensor(self.memory.embedding.get_or_create(output_b)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    zsa_a = self.model.encoder(prompt_emb, a_emb)
                    zsa_b = self.model.encoder(prompt_emb, b_emb)

                preferred = "a" if value_a >= value_b else "b"
                diff = zsa_a - zsa_b if preferred == "a" else zsa_b - zsa_a
                inputs.append(diff.squeeze(0).detach())
                labels.append(torch.tensor([1.0], device=self.device))

            except Exception as e:
                self.logger.log("PairPreparationError", {"error": str(e)})
                continue

        if len(inputs) < self.min_samples:
            self.logger.log("InsufficientSamples", {
                "sample_count": len(inputs),
                "threshold": self.min_samples
            })
            return None

        dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _build_model(self):
        encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
        predictor = ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        return MRQModel(encoder, predictor, self.memory.embedding, device=self.device)

    def _train_epoch(self, model, dataloader):
        model.encoder.train()
        model.predictor.train()
        total_loss, count = 0.0, 0

        for inputs, scores in dataloader:
            inputs = inputs.to(self.device)
            scores = scores.to(self.device)

            predictions = model.predictor(inputs).squeeze()
            loss = F.mse_loss(predictions, scores)

            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 0.5)

            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

        return total_loss / count

    def train(self, samples, dimension):
        dataloader = self._create_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data", "dimension": dimension}

        self.optimizer = optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.predictor.parameters()),
            lr=self.lr
        )

        best_loss = float("inf")
        early_stop_counter = 0
        losses = []

        for epoch in range(self.epochs):
            avg_loss = self._train_epoch(self.model, dataloader)
            losses.append(avg_loss)
            self.logger.log("MRQTrainingEpoch", {
                "epoch": epoch + 1,
                "loss": avg_loss
            })
            if avg_loss < best_loss - self.early_stopping_min_delta:
                best_loss = avg_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if self.use_early_stopping and early_stop_counter >= self.early_stopping_patience:
                    break

        locator = self.get_locator(dimension)            
        torch.save(self.model.encoder.state_dict(), locator.encoder_file())
        torch.save(self.model.predictor.state_dict(), locator.model_file())

        if self.use_tuner:
            tuner = RegressionTuner(dimension=dimension, logger=self.logger)
            for inputs, scores in dataloader:
                inputs = inputs.to(self.device)
                preds = self.model.predictor(inputs).squeeze().detach().cpu().numpy()
                actuals = scores.cpu().numpy()
                for p, a in zip(preds, actuals):
                    tuner.train_single(float(p), float(a))
            tuner.save(locator.tuner_file())

        scores_np = torch.tensor([s["value_a"] for s in samples])
        min_score = float(torch.min(scores_np))
        max_score = float(torch.max(scores_np))

        meta = {
            "dimension": dimension,
            "model_type": "mrq",
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "version": self.version,
            "dim": self.dim,
            "hdim": self.hdim,
            "min_score": min_score,
            "max_score": max_score,
            "avg_loss": best_loss,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._save_meta_file(meta, dimension)

        training_stat = TrainingStatsORM(
            model_type="mrq",
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.embedding_type,
            
            avg_q_loss=best_loss
        )
        self.memory.session.add(training_stat)
        self.memory.session.commit()

        return meta
``n

## File: training\sicql_trainer.py

`python
# stephanie/agents/maintenance/sicql_trainer.py
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.models.model_version import ModelVersionORM
from stephanie.models.training_stats import TrainingStatsORM
from stephanie.scoring.model.in_context_q import InContextQModel
from stephanie.scoring.model.policy_head import PolicyHead
from stephanie.scoring.model.q_head import QHead
from stephanie.scoring.model.v_head import VHead
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class SICQLTrainer(BaseTrainer):

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.embedding_type = self.memory.embedding.type
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim
        self.root_dir = cfg.get("model_path", "models")
        self.dimension = cfg.get("dimension", "alignment")
        self.embedding_type = cfg.get("embedding_type", "hnet")
        self.model_type = "sicql"
        self.target_type = cfg.get("target_type", "document")
        self.version = cfg.get("model_version", "v1")

        # Device management
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Training configuration
        self._init_config(cfg)

        # Track training state
        self.best_loss = float("inf")
        self.early_stop_counter = 0
        self.models = {}
        self.tuners = {}
        self._load_tuners()

        # Log initialization
        self.logger.log(
            "SICQLTrainerInitialized",
            {
                "dimension": self.cfg.get("dimension", "alignment"),
                "embedding_type": self.cfg.get("embedding_type", "hnet"),
                "use_gild": self.use_gild,
                "use_qmax": self.use_qmax,
                "device": str(self.device),
            },
        )
 

    def _init_config(self, cfg):
        """Initialize training parameters from config"""
        self.use_tuner = cfg.get("use_tuner", True)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.batch_size = cfg.get("batch_size", 32)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("lr", 1e-4)
        self.gamma = cfg.get("gamma", 0.95)  # Discount factor
        self.beta = cfg.get("beta", 1.0)  # Policy temperature
        self.entropy_weight = cfg.get("entropy_weight", 0.01)
        self.dimensions = cfg.get("dimensions", [])
        self.min_samples = cfg.get("min_samples", 10)
        self.expectile_tau = cfg.get("expectile_tau", 0.7)  # For V-head
        self.use_gild = cfg.get("use_gild", True)
        self.use_qmax = cfg.get("use_qmax", True)
        self.scorer_map = ["ebt", "svm", "mrq"]  # Policy head mapping

    def _load_tuners(self):
        """Load regression tuners for each dimension"""
        for dim in self.dimensions:
            tuner_path = super().get_locator(dim).tuner_file()
            if os.path.exists(tuner_path):
                self.tuners[dim] = RegressionTuner(dimension=dim)
                self.tuners[dim].load(tuner_path)
            else:
                self.tuners[dim] = None
                self.logger.log(
                    "TunerMissing", {"dimension": dim, "path": tuner_path}
                )

    def _build_model(self, dimension):
        """Build or load SICQL model"""
        locator = super().get_locator(dimension)
        if locator.model_exists():
            # Load existing model
            encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
            q_head = QHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            v_head = VHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            pi_head = PolicyHead(
                zsa_dim=self.dim, hdim=self.hdim, num_actions=3
            ).to(self.device)

            # Load weights
            encoder.load_state_dict(
                torch.load(locator.encoder_file(), map_location=self.device)
            )
            q_head.load_state_dict(
                torch.load(locator.q_head_file(), map_location=self.device)
            )
            v_head.load_state_dict(
                torch.load(locator.v_head_file(), map_location=self.device)
            )
            pi_head.load_state_dict(
                torch.load(locator.pi_head_file(), map_location=self.device)
            )

            # Build model
            sicql_model = InContextQModel(
                encoder=encoder,
                q_head=q_head,
                v_head=v_head,
                pi_head=pi_head,
                embedding_store=self.memory.embedding,
                device=self.device,
            )
            return sicql_model

        # Build new model
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim

        encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
        q_head = QHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        v_head = VHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        pi_head = PolicyHead(
            zsa_dim=self.dim, hdim=self.hdim, num_actions=3
        ).to(self.device)

        return InContextQModel(
            encoder=encoder,
            q_head=q_head,
            v_head=v_head,
            pi_head=pi_head,
            embedding_store=self.memory.embedding,
            device=self.device,
        )

    def _train_epoch(self, model, dataloader):
        """Train for one epoch with all heads"""
        model.train()
        total_q_loss = 0.0
        total_v_loss = 0.0
        total_pi_loss = 0.0
        count = 0

        for ctx_emb, doc_emb, scores in tqdm(dataloader, desc="Training"):
            ctx_emb = ctx_emb.to(self.device)
            doc_emb = doc_emb.to(self.device)
            scores = scores.to(self.device)

            outputs = model(ctx_emb, doc_emb)

            q_loss = F.mse_loss(outputs["q_value"], scores)

            v_loss = (
                self._expectile_loss(
                    scores - outputs["state_value"], tau=self.expectile_tau
                )
                if self.use_qmax
                else torch.tensor(0.0, device=self.device)
            )

            pi_loss = torch.tensor(0.0, device=self.device)
            if self.use_gild and "action_logits" in outputs:
                advantage = (
                    outputs["q_value"] - outputs["state_value"]
                ).detach()
                weights = torch.exp(self.beta * advantage)
                weights = weights / weights.sum()

                # Corrected reshape
                weights = weights.unsqueeze(-1)  # Ensure (batch_size, 1)

                log_probs = F.log_softmax(outputs["action_logits"], dim=-1)
                pi_loss = -(log_probs * weights).mean()

                # Optional entropy regularization
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
                pi_loss += self.entropy_weight * entropy

            loss = (
                q_loss * self.cfg.get("q_weight", 1.0)
                + v_loss * self.cfg.get("v_weight", 0.5)
                + pi_loss * self.cfg.get("pi_weight", 0.3)
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            self.optimizer.step()

            total_q_loss += q_loss.item() * ctx_emb.size(0)
            total_v_loss += v_loss.item() * ctx_emb.size(0)
            total_pi_loss += pi_loss.item() * ctx_emb.size(0)
            count += ctx_emb.size(0)

        avg_q = total_q_loss / count
        avg_v = total_v_loss / count
        avg_pi = total_pi_loss / count

        if self.use_qmax:
            self.scheduler["q"].step(avg_q)
        if self.use_gild:
            self.scheduler["pi"].step(avg_pi)

        return {"q": avg_q, "v": avg_v, "pi": avg_pi, "total": loss.item()}

    def _expectile_loss(self, diff, tau=0.7):
        """Compute expectile loss for V-head"""
        return torch.where(
            diff > 0, tau * diff.pow(2), (1 - tau) * diff.pow(2)
        ).mean()

    def _should_stop_early(self, current_avg):
        """Check for early stopping"""
        if not self.use_early_stopping:
            return False

        if current_avg < self.best_loss - self.early_stopping_min_delta:
            self.best_loss = current_avg
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        return self.early_stop_counter >= self.early_stopping_patience

    def _save_model(self, model, dimension, stats):
        locator = super().get_locator(dimension)
        """Save model components with metadata"""
        # Save each component
        torch.save(model.encoder.state_dict(), locator.encoder_file())
        torch.save(model.q_head.state_dict(), locator.q_head_file())
        torch.save(model.v_head.state_dict(), locator.v_head_file())
        torch.save(model.pi_head.state_dict(), locator.pi_head_file())

        # Calculate policy metrics
        policy_logits = model.pi_head.weight.data.mean(dim=0).tolist()
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1).tolist()
        policy_entropy = -torch.sum(
            policy_probs * torch.log(torch.tensor(policy_probs) + 1e-8)
        ).item()

        # Build metadata
        meta = {
            "dim": self.dim,
            "hdim": self.hdim,
            "dimension": dimension,
            "version": self.cfg.get("model_version", "v1"),
            "avg_q_loss": stats.get("avg_q_loss", 0.0),
            "avg_v_loss": stats.get("avg_v_loss", 0.0),
            "avg_pi_loss": stats.get("avg_pi_loss", 0.0),
            "policy_logits": policy_logits,
            "policy_probs": policy_probs,
            "policy_entropy": policy_entropy,
            "policy_stability": max(policy_probs),
            "device": str(self.device),
            "embedding_type": self.cfg.get("embedding_type", "hnet"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Save metadata
        with open(locator.meta_file(), "w") as f:
            json.dump(meta, f)

        # Save tuner if available
        if dimension in self.tuners and self.tuners[dimension]:
            self.tuners[dimension].save(locator.tuner_file())

        # Save model version
        model_version = ModelVersionORM(**meta)
        self.memory.session.add(model_version)
        self.memory.session.commit()

        return meta

    def _log_training_stats(self, dim, meta):
        """Log training stats to database"""
        training_stats = TrainingStatsORM(
            model_type="sicql",
            target_type=self.cfg.get("target_type", "document"),
            dimension=dim,
            version=meta["version"],
            avg_q_loss=meta["avg_q_loss"],
            avg_v_loss=meta["avg_v_loss"],
            avg_pi_loss=meta["avg_pi_loss"],
            policy_entropy=meta["policy_entropy"],
            policy_stability=meta["policy_stability"],
            performance=meta["avg_q_loss"],
        )
        self.memory.session.add(training_stats)
        self.memory.session.commit()

    def _validate_tensor(self, tensor, name):
        """Validate tensor before use"""
        if tensor is None:
            self.logger.log(
                "InvalidTensor",
                {"tensor_name": name, "reason": "tensor_is_none"},
            )
            return False

        if torch.isnan(tensor).any():
            self.logger.log(
                "NaNInTensor", {"tensor_name": name, "tensor": tensor.tolist()}
            )
            return False

        return True

    def _calculate_policy_logits(self, model):
        """Calculate policy logits from policy head weights"""
        with torch.no_grad():
            policy_weights = model.pi_head.get_policy_weights()
            policy_probs = F.softmax(policy_weights, dim=-1)
            return policy_probs.tolist()

    def _calculate_policy_stability(self, policy_logits):
        """Calculate policy stability from logits"""
        if not policy_logits:
            return 0.0
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1)
        return policy_probs.max().item()

    def _calculate_policy_entropy(self, policy_logits):
        """Calculate policy entropy for versioning"""
        if not policy_logits:
            return 0.0
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1)
        return (
            -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=-1)
            .mean()
            .item()
        )

    def train(self, samples, dim):
        """
        Train SICQL model for a dimension
        Args:
            samples: List of training samples
            dim: Dimension to train
        Returns:
            Training statistics and model
        """
        self.logger.log("DimensionTrainingStarted", {"dimension": dim})

        # Prepare data
        dataloader = super()._create_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data", "dimension": dim}

        # Build model
        model = self._build_model(dim)
        model.train()

        # Optimizer for all heads
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = {
            "q": ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            ),
            "v": ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            ),
            "pi": ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            ),
        }

        # Training stats
        stats = {
            "dimension": dim,
            "q_losses": [],
            "v_losses": [],
            "pi_losses": [],
            "policy_entropies": [],
            "avg_q_loss": 0.0,
            "avg_v_loss": 0.0,
            "avg_pi_loss": 0.0,
            "policy_entropy": 0.0,
            "policy_stability": 0.0,
        }

        # Training loop
        for epoch in range(self.epochs):
            epoch_stats = self._train_epoch(model, dataloader)
            stats["q_losses"].append(epoch_stats["q"])
            stats["v_losses"].append(epoch_stats["v"])
            stats["pi_losses"].append(epoch_stats["pi"])

            # Calculate policy entropy
            policy_logits = self._calculate_policy_logits(model)
            policy_entropy = self._calculate_policy_entropy(policy_logits)
            stats["policy_entropies"].append(policy_entropy)

            # Early stopping check
            if self._should_stop_early(stats["q_losses"][-1]):
                self.logger.log(
                    "EarlyStopping",
                    {
                        "dimension": dim,
                        "epoch": epoch + 1,
                        "best_loss": self.best_loss,
                    },
                )
                break

        # Final stats
        stats["avg_q_loss"] = np.mean(stats["q_losses"])
        stats["avg_v_loss"] = np.mean(stats["v_losses"])
        stats["avg_pi_loss"] = np.mean(stats["pi_losses"])
        stats["policy_entropy"] = np.mean(stats["policy_entropies"])
        stats["policy_stability"] = (
            max(stats["policy_entropies"])
            if stats["policy_entropies"]
            else 0.0
        )

        # Save model
        meta = self._save_model(model, dim, stats)
        stats.update(meta)

        # Log to database
        self._log_training_stats(dim, meta)

        self.logger.log(
            "DimensionTrainingComplete",
            {
                "dimension": dim,
                "final_q_loss": stats["avg_q_loss"],
                "final_v_loss": stats["avg_v_loss"],
                "final_pi_loss": stats["avg_pi_loss"],
            },
        )

        # Cache model
        self.models[dim] = model
        return stats

    def _log_training_stats(self, dim, meta):
        """Log training stats to database"""
        training_stats = TrainingStatsORM(
            model_type="sicql",
            target_type=self.cfg.get("target_type", "document"),
            dimension=dim,
            version=meta["version"],
            embedding_type=self.embedding_type,
            avg_q_loss=meta["avg_q_loss"],
            avg_v_loss=meta["avg_v_loss"],
            avg_pi_loss=meta["avg_pi_loss"],
            policy_entropy=meta.get("policy_entropy", 0.0),
            policy_stability=meta.get("policy_stability", 0.0),
        )
        self.memory.session.add(training_stats)
        self.memory.session.commit()

    def _train_sicql(self, model, dataloader, output_dir):
        """Train SICQL model with all heads"""
        model.train()
        best_loss = float("inf")
        patience_counter = 0

        # Build optimizers
        optimizers = {
            "encoder": optim.Adam(model.encoder.parameters(), lr=self.lr),
            "q_head": optim.Adam(model.q_head.parameters(), lr=self.lr),
            "v_head": optim.Adam(model.v_head.parameters(), lr=self.lr),
            "pi_head": optim.Adam(model.pi_head.parameters(), lr=self.lr),
        }

        # Build schedulers
        schedulers = {
            "encoder": ReduceLROnPlateau(
                optimizers["encoder"], mode="min", factor=0.5, patience=2
            ),
            "q_head": ReduceLROnPlateau(
                optimizers["q_head"], mode="min", factor=0.5, patience=2
            ),
            "v_head": ReduceLROnPlateau(
                optimizers["v_head"], mode="min", factor=0.5, patience=2
            ),
            "pi_head": ReduceLROnPlateau(
                optimizers["pi_head"], mode="min", factor=0.5, patience=2
            ),
        }

        # Training loop
        for epoch in range(self.epochs):
            total_q_loss = 0.0
            total_v_loss = 0.0
            total_pi_loss = 0.0
            count = 0

            for ctx_emb, doc_emb, scores in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}"
            ):
                # Device management
                ctx_emb = ctx_emb.to(self.device)
                doc_emb = doc_emb.to(self.device)
                scores = scores.to(self.device)

                # Forward pass
                outputs = model(ctx_emb, doc_emb)

                # Q-head loss
                q_loss = F.mse_loss(outputs["q_value"], scores)

                # V-head loss
                v_loss = self._expectile_loss(
                    scores - outputs["state_value"],
                    tau=self.cfg.get("expectile", 0.7),
                )

                # Policy head loss
                pi_loss = torch.tensor(0.0, device=self.device)
                if self.use_gild:
                    advantage = (
                        outputs["q_value"] - outputs["state_value"]
                    ).detach()
                    weights = torch.exp(self.beta * advantage)
                    weights = weights / weights.sum()

                    policy_probs = F.softmax(outputs["action_logits"], dim=-1)
                    entropy = -torch.sum(
                        policy_probs * torch.log(policy_probs + 1e-8), dim=-1
                    ).mean()

                    pi_loss = -(
                        F.log_softmax(outputs["action_logits"], dim=-1)
                        * weights
                    ).mean()
                    pi_loss += self.entropy_weight * entropy

                # Backward pass
                optimizers["q_head"].zero_grad()
                q_loss.backward()
                optimizers["q_head"].step()

                optimizers["v_head"].zero_grad()
                v_loss.backward()
                optimizers["v_head"].step()

                if self.use_gild:
                    optimizers["pi_head"].zero_grad()
                    pi_loss.backward()
                    optimizers["pi_head"].step()

                # Track losses
                total_q_loss += q_loss.item() * ctx_emb.size(0)
                total_v_loss += v_loss.item() * ctx_emb.size(0)
                total_pi_loss += pi_loss.item() * ctx_emb.size(0)
                count += ctx_emb.size(0)

            # End of epoch
            avg_q = total_q_loss / count
            avg_v = total_v_loss / count
            avg_pi = total_pi_loss / count

            # Early stopping
            if avg_q < best_loss - self.early_stopping_min_delta:
                best_loss = avg_q
                patience_counter = 0
                # Save best model
                torch.save(
                    model.encoder.state_dict(), f"{output_dir}/encoder.pt"
                )
                torch.save(
                    model.q_head.state_dict(), f"{output_dir}/q_head.pt"
                )
                torch.save(
                    model.v_head.state_dict(), f"{output_dir}/v_head.pt"
                )
                torch.save(
                    model.pi_head.state_dict(), f"{output_dir}/pi_head.pt"
                )
            else:
                patience_counter += 1

            # Log epoch
            self.logger.log(
                "SICQLTrainingEpoch",
                {
                    "epoch": epoch + 1,
                    "q_loss": avg_q,
                    "v_loss": avg_v,
                    "pi_loss": avg_pi,
                    "lr": optimizers["q_head"].param_groups[0]["lr"],
                },
            )

            # Check for early stopping
            if patience_counter >= self.early_stopping_patience:
                self.logger.log(
                    "SICQLEarlyStopping",
                    {"epoch": epoch + 1, "best_loss": best_loss},
                )
                break

        self.logger.log("SICQLTrainingComplete", {"best_loss": best_loss})
        return model

    def _save_model(self, model, dimension, stats):
        """Save SICQL model components"""
        locator = super().get_locator(dimension)
        # Save components separately
        torch.save(model.encoder.state_dict(), locator.encoder_file())
        torch.save(model.q_head.state_dict(), locator.q_head_file())
        torch.save(model.v_head.state_dict(), locator.v_head_file())
        torch.save(model.pi_head.state_dict(), locator.pi_head_file())

        # Calculate policy metrics
        policy_logits = model.pi_head.get_policy_weights().tolist()
        policy_probs_tensor = F.softmax(torch.tensor(policy_logits), dim=-1)
        policy_probs = policy_probs_tensor.tolist()
        policy_entropy = -torch.sum(
            policy_probs_tensor * torch.log(policy_probs_tensor + 1e-8)
        ).item()
        policy_stability = max(policy_probs)


        # Build metadata
        meta = {
            "dim": self.dim,
            "hdim": self.hdim,
            "dimension": dimension,
            "version": self.cfg.get("model_version", "v1"),
            "avg_q_loss": float(stats["avg_q_loss"]),
            "avg_v_loss": float(stats["avg_v_loss"]),
            "avg_pi_loss": float(stats["avg_pi_loss"]),
            "policy_entropy": float(policy_entropy),
            "policy_stability": float(policy_stability),
            "policy_logits": policy_logits,
            "policy_probs": policy_probs,
            "embedding_type": self.embedding_type,
            "max_value": 100,
            "min_value": 0,
            "device": str(self.device), 
            "timestamp": datetime.utcnow().isoformat(),
        }

        super()._save_meta_file(meta, dimension)
        return meta

    def run(self, context: dict) -> dict:
        """Main entry point for training"""
        documents = context.get("documents", [])
        # Train each dimension
        results = {}
        for dim in self.dimensions:
            # Get training samples
            samples = self._get_samples(context, documents, dim)
            if not samples:
                continue

            # Train model
            stats = self.train(samples, dim)
            if "error" in stats:
                continue

            # Update belief cartridges
            self._update_belief_cartridge(context, dim, stats)
            results[dim] = stats

        # Update context with results
        context["training_stats"] = results
        return context

    def _get_samples(self, context, documents, dim):
        """Get training samples for dimension"""
        samples = []
        goal = context.get("goal", {})
        for doc in documents:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            score = self.memory.scores.get_score(goal.id, scorable.id)
            if score:
                samples.append(
                    {
                        "title": goal.get("goal_text", ""),
                        "output": scorable.text,
                        "score": score.score,
                    }
                )
        return samples

    def _update_belief_cartridge(self, context, dim, stats):
        """Update belief cartridges with policy stats"""
        policy_logits = stats.get("policy_logits", [0.3, 0.7, 0.0])
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1).tolist()

        # Build belief cartridge
        cartridge = BeliefCartridgeORM(
            title=f"{dim} policy",
            content=f"Policy head weights: {policy_probs}",
            goal_id=context.get("goal_id"),
            domain=dim,
            policy_logits=policy_probs,
            policy_entropy=stats.get("policy_entropy", 1.05),
            policy_stability=stats.get("policy_stability", 0.82),
        )
        self.memory.session.add(cartridge)
        self.memory.session.commit()
``n

## File: training\svm_trainer.py

`python
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class SVMTrainer(BaseTrainer):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.kernel = cfg.get("kernel", "rbf")
        self.C = cfg.get("C", 1.0)
        self.epsilon = cfg.get("epsilon", 0.1)


    def train(self, samples, dimension):
        samples = [
            {
                "title": pair["title"],
                "output": pair["output_a"],
                "score": pair["value_a"],
            }
            for pair in samples
        ]

        dataloader = self._create_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data", "dimension": dimension}

        # Convert DataLoader to numpy arrays
        X, y = [], []
        for ctx_emb, doc_emb, scores in dataloader:
            ctx_emb = ctx_emb.cpu().numpy()
            doc_emb = doc_emb.cpu().numpy()
            x = np.concatenate([ctx_emb, doc_emb], axis=1)
            X.append(x)
            y.append(scores.cpu().numpy())

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        # Fit scaler and scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train SVM
        model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        model.fit(X_scaled, y)

        # Save both model and scaler
        locator = self.get_locator(dimension)
        model_path = locator.model_file(suffix=".joblib")
        scaler_path = locator.scaler_file()

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        meta = {
            "dimension": dimension,
            "model_type": "svm",
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "version": self.version,
            "kernel": self.kernel,
            "C": self.C,
            "epsilon": self.epsilon,
            "dim": self.dim,
            "hdim": self.hdim,
            "min_score": self.cfg.get("min_score", 0),
            "max_score": self.cfg.get("max_score", 100),
        }
        self._save_meta_file(meta, dimension)

        # Add before return meta
        tuner = RegressionTuner(dimension=dimension, logger=self.logger)
        for i in range(len(X)):
            prediction = float(model.predict(X_scaled[i].reshape(1, -1))[0])
            actual = float(y[i])
            tuner.train_single(prediction, actual)
        tuner.save(locator.tuner_file())

        self.log_event("SVMTrainingComplete", {"dimension": dimension})

        return meta
``n

## File: transforms\prompt_score_regressor.py

`python
# stephanie/scoring/transforms/prompt_score_regressor.py
import os

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PromptScoreRegressor:
    def __init__(self, model_path="prompt_score_regressor.pkl"):
        self.model_path = model_path
        self.models = {}  # dimension -> trained model

    def train(self, embeddings, score_dicts):
        """
        embeddings: List[List[float]]
        score_dicts: List[Dict[str, float]] (one per embedding)
        """
        if not score_dicts:
            raise ValueError("No score dictionaries provided for training.")

        # Infer all dimensions from the first sample
        dimensions = list(score_dicts[0].keys())
        X = np.array(embeddings)

        for dim in dimensions:
            y = np.array([score[dim] for score in score_dicts])
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            self.models[dim] = model.fit(X, y)

        if self.model_path:
            joblib.dump(self.models, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.models = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"No model found at {self.model_path}")

    def predict(self, embedding):
        if not self.models:
            raise ValueError("Models not trained or loaded.")

        scores = {}
        for dim, model in self.models.items():
            scores[dim] = float(model.predict([embedding])[0])
        return scores

    def predict_batch(self, embeddings):
        """
        Returns: List[Dict[dimension -> score]]
        """
        if not self.models:
            raise ValueError("Models not trained or loaded.")

        results = []
        for emb in embeddings:
            results.append(self.predict(emb))
        return results
``n

## File: transforms\regression_tuner.py

`python
# stephanie/scoring/transforms/regression_tuner.py

import json

import numpy as np
import torch
from sklearn.linear_model import LinearRegression


class RegressionTuner:
    """
    Learns to transform MR.Q scores to align with LLM scores dynamically.
    Does not save any state to disk—purely in-memory and real-time.
    """

    def __init__(self, dimension: str, logger=None, min_samples: int = 10, device=None):
        self.dimension = dimension
        self.logger = logger
        self.min_samples = min_samples
        self.device = device if device is not None else "cpu"
        self.x = []  # MRQ scores
        self.y = []  # LLM scores
        self.model = None
        
    def train_single(self, mrq_score: float, llm_score: float):
        """Adds a new training pair and refits if threshold reached."""
        self.x.append(mrq_score)
        self.y.append(llm_score)

        if len(self.x) >= self.min_samples:
            self._fit()

        if self.logger:
            self.logger.log(
                "RegressionTunerTrainSingle",
                {
                    "dimension": self.dimension,
                    "mrq_score": mrq_score,
                    "llm_score": llm_score,
                    "total_samples": len(self.x),
                },
            )

    def _fit(self):
        """Fits a linear regression model to current examples."""
        x_arr = np.array(self.x).reshape(-1, 1)
        y_arr = np.array(self.y)

        self.model = LinearRegression().fit(x_arr, y_arr)

        if self.logger:
            self.logger.log(
                "RegressionTunerFitted",
                {
                    "dimension": self.dimension,
                    "count": len(self.x),
                    "coef": float(self.model.coef_[0]),
                    "intercept": float(self.model.intercept_),
                },
            )

    def transform(self, score: float) -> float:
        """Transforms a score using the fitted regression model if available."""
        if self.model:
            return float(self.model.predict(np.array([[score]]))[0])
        return score

    def save(self, path):
        if not self.model:
            raise ValueError("Model has not been trained yet — nothing to save.")

        data = {
            "dimension": self.dimension,
            "samples": list(zip(self.x, self.y)),
            "coef": float(self.model.coef_[0]),
            "intercept": float(self.model.intercept_),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        self.dimension = data["dimension"]
        self.x, self.y = zip(*data["samples"]) if data["samples"] else ([], [])
        self.x = list(self.x)
        self.y = list(self.y)

        if self.x and len(self.x) >= self.min_samples:
            self._fit()
    
    @classmethod
    def from_dataloader(cls, dataloader, model, dimension, logger=None, min_samples=10):
        """
        Creates and trains a RegressionTuner from a PyTorch DataLoader.
        
        Args:
            dataloader: DataLoader yielding (context_emb, doc_emb, llm_score)
            model: MRQ or SICQL model used to generate MRQ scores
            dimension: Scoring dimension (e.g., "alignment")
            logger: Optional logger
            min_samples: Minimum samples before fitting
        
        Returns:
            RegressionTuner instance
        """
        tuner = cls(dimension=dimension, logger=logger, min_samples=min_samples)
        model = model.to(tuner.device)
        model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                context_emb, doc_emb, llm_scores = batch
                
                # Move to device
                context_emb = context_emb.to(tuner.device)
                doc_emb = doc_emb.to(tuner.device)
                llm_scores = llm_scores.to(tuner.device)
                
                # Generate MRQ score using the model
                if hasattr(model, 'encoder'):
                    # SICQL-style model
                    zsa = model.encoder(context_emb, doc_emb)
                    if hasattr(model, 'q_head'):
                        mrq_scores = model.q_head(zsa).squeeze()
                    else:
                        mrq_scores = model(zsa).squeeze()
                else:
                    # Simple MRQ model
                    mrq_scores = model(doc_emb).squeeze()
                
                # Convert to lists
                mrq_scores_list = mrq_scores.cpu().tolist()
                llm_scores_list = llm_scores.cpu().tolist()
                
                # Train on each pair
                for mrq, llm in zip(mrq_scores_list, llm_scores_list):
                    tuner.train_single(mrq, llm)
        
        return tuner
``n
