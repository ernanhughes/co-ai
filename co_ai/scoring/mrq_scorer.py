import torch

from co_ai.evaluator.hypothesis_value_predictor import HypothesisValuePredictor
from co_ai.evaluator.mrq_trainer import MRQTrainer
from co_ai.evaluator.text_encoder import TextEncoder
from co_ai.models.sharpening_prediction import SharpeningPredictionORM
from co_ai.scoring.base_scorer import BaseScorer
from co_ai.scoring.score_result import ScoreResult
from co_ai.scoring.score_bundle import ScoreBundle
from co_ai.scoring.scoring_manager import ScoringManager


class MRQScorer(BaseScorer):
    def __init__(self, cfg: dict, memory, logger, dimensions=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.device = cfg.get("device", "cpu")

        self.dimensions = dimensions or ["mrq"]
        self.models = {}  # dimension -> (encoder, predictor)
        self.trainers = {}  # dimension -> MRQTrainer
        self.min_score_by_dim = {}
        self.max_score_by_dim = {}
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)
        self.encoder = TextEncoder().to(self.device)

        for dim in self.dimensions:
            trainer = MRQTrainer(
                memory=self.memory,
                logger=self.logger,
                value_predictor=self.value_predictor,
                encoder=self.encoder,
                device=self.device,
            )
            self.models[dim] = (self.encoder, self.value_predictor)
            self.trainers[dim] = trainer
            self.min_score_by_dim[dim] = 0.0
            self.max_score_by_dim[dim] = 1.0

    def score(self, goal: dict, hypothesis: dict, dimensions: list[str]) -> ScoreBundle:
        """
        Scores a hypothesis using MR.Q on the specified dimensions.
        Returns a dictionary mapping dimension names to ScoreResult instances.
        """
        results = []
        for dim in dimensions:
            score = self._estimate_score(goal, hypothesis, dim)
            rationale = f"MRQ estimated score for {dim}."

            self.logger.log(
                "MRQDimensionEvaluated",
                {"dimension": dim, "score": score, "rationale": rationale},
            )

            results.append(ScoreResult(
                dimension=dim,
                score=score,
                rationale=rationale,
                weight=1.0,
                source="mrq",
            ))

        final_score = round(sum(d.score for d in results) / len(results), 2)

        bundle = ScoreBundle(results={r.dimension: r for r in results})
        ScoringManager.save_score_to_memory(
            bundle,
            hypothesis,
            cfg=self.cfg,
            memory=self.memory,
            logger=self.logger,
            source="mrq",
        )
        return bundle

    def _estimate_score(self, goal: dict, hypothesis: dict, dimension: str) -> float:
        print(f"Estimating score for dimension: {dimension}")
        if dimension not in self.models:
            raise ValueError(f"Dimension '{dimension}' not found in MRQ models.")
        encoder, predictor = self.models[dimension]

        prompt_text = goal.get("prompt") or goal.get("text") or ""
        response_text = hypothesis.get("response") or hypothesis.get("text") or ""

        # Get embeddings
        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt_text), device=self.device
        ).unsqueeze(0)

        response_emb = torch.tensor(
            self.memory.embedding.get_or_create(response_text), device=self.device
        ).unsqueeze(0)

        # Encode and predict value
        zsa = encoder(prompt_emb, response_emb)
        raw_score = predictor(zsa).item()

        # Normalize to 0–100 scale
        norm_score = self.normalize_score(raw_score, dimension)
        return norm_score

    def evaluate(self, prompt: str, response: str) -> ScoreBundle:
        # Evaluate each dimension using the trained models
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

        final_score = round(sum(d.score for d in results) / len(results), 2)

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

    def normalize_score(self, raw, dim):
        min_ = self.min_score_by_dim.get(dim, 0.0)
        max_ = self.max_score_by_dim.get(dim, 1.0)
        range_ = (max_ - min_) or 1.0
        return round(100 * (raw - min_) / range_, 2)

    def judge(self, goal, prompt, output_a, output_b):
        # Optional: Per-dimension judging could be added later
        dim = self.dimensions[0]
        encoder, predictor = self.models[dim]

        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)
        output_a_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_a), device=self.device
        ).unsqueeze(0)
        output_b_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_b), device=self.device
        ).unsqueeze(0)

        zsa_a = encoder(prompt_emb, output_a_emb)
        zsa_b = encoder(prompt_emb, output_b_emb)

        value_a = predictor(zsa_a).item()
        value_b = predictor(zsa_b).item()

        preferred_output = output_a if value_a >= value_b else output_b

        if self.memory.mrq.log_evaluations():
            prediction = SharpeningPredictionORM(
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
            self.memory.sharpening.insert_sharpening_prediction(
                prediction.to_dict(), goal
            )

        return preferred_output, {"value_a": value_a, "value_b": value_b}

    def train_from_database(self, cfg: dict):
        all_samples = self.memory.mrq.get_training_pairs_by_dimension()

        for dim, samples in all_samples.items():
            if not samples:
                continue

            # Dynamically create trainer for unseen dimension
            if dim not in self.trainers:
                self.trainers[dim] = MRQTrainer(
                    memory=self.memory,
                    logger=self.logger,
                    value_predictor=self.value_predictor,
                    encoder=self.encoder,
                    device=self.device,
                )

            self.update_score_bounds_from_data(samples, dim)
            trainer = self.trainers[dim]
            dataloader = trainer.prepare_training_data(samples)
            trainer.train(dataloader, cfg)

    def train_from_context(self, context: dict, cfg: dict):
        dim_samples = context.get("mrq_training_pairs_by_dimension", {})

        for dim, samples in dim_samples.items():
            if not samples:
                continue
            self.update_score_bounds_from_data(samples, dim)
            trainer = self.trainers[dim]
            dataloader = trainer.prepare_training_data(samples)
            trainer.train(dataloader, cfg)

    def update_score_bounds_from_data(self, samples: list, dim: str):
        values = []
        for sample in samples:
            if "value_a" in sample and "value_b" in sample:
                values.extend([sample["value_a"], sample["value_b"]])
            elif "value" in sample:
                values.append(sample["value"])

        if values:
            self.min_score_by_dim[dim] = min(values)
            self.max_score_by_dim[dim] = max(values)
            self.logger.log(
                "MRQScoreBoundsUpdated",
                {
                    "dimension": dim,
                    "min_score": self.min_score_by_dim[dim],
                    "max_score": self.max_score_by_dim[dim],
                    "count": len(values),
                },
            )
