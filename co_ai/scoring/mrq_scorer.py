import torch

from co_ai.evaluator.hypothesis_value_predictor import HypothesisValuePredictor
from co_ai.evaluator.mrq_trainer import MRQTrainer
from co_ai.evaluator.text_encoder import TextEncoder
from co_ai.models.sharpening_prediction import SharpeningPredictionORM
from co_ai.scoring.base_scorer import BaseScorer
from co_ai.scoring.score_result import ScoreResult
from co_ai.scoring.score_bundle import ScoreBundle
from co_ai.scoring.scoring_manager import ScoringManager

from co_ai.scoring.transforms.regression_tuner import RegressionTuner

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
        self.regression_tuners = {}

        for dim in self.dimensions:
            self.regression_tuners[dim] = RegressionTuner(
                dimension=dim, logger=self.logger
            )
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
        bundle = ScoreBundle(results={r.dimension: r for r in results})
        return bundle

    def _estimate_score(self, goal: dict, hypothesis: dict, dimension: str) -> float:
        print(f"Estimating score for dimension: {dimension}")
        if dimension not in self.models:
            self.regression_tuners[dimension] = RegressionTuner(
                dimension=dimension, logger=self.logger
            )
            self.logger.log("MRQModelInitializing", {"dimension": dimension})
            trainer = MRQTrainer(
                memory=self.memory,
                logger=self.logger,
                value_predictor=self.value_predictor,
                encoder=self.encoder,
                device=self.device,
            )
            self.models[dimension] = (self.encoder, self.value_predictor)
            self.trainers[dimension] = trainer
            self.min_score_by_dim[dimension] = 0.0
            self.max_score_by_dim[dimension] = 1.0
        encoder, predictor = self.models[dimension]

        prompt_text = goal.get("goal_text")
        response_text = hypothesis.get("text")

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
        tuner = self.regression_tuners.get(dimension)
        if tuner:
            tuned_score = tuner.transform(norm_score)
            self.logger.log(
                "MRQTunedScore",
                {
                    "dimension": dimension,
                    "raw": norm_score,
                    "tuned": tuned_score,
                },
            )
            return tuned_score
        return norm_score

    def align_to_best_llm_neighbour(self, goal: dict, hypothesis: dict, dimension: str):
        llm_scores = self.get_closest_llm_scores(hypothesis["text"], dimension)
        if not llm_scores:
            return  # no alignment possible
        best_llm_score = max(llm_scores)
        self.align_with_llm_score(dimension, goal, hypothesis, best_llm_score)

    def get_closest_llm_scores(self, hypothesis_text: str, dimension: str, top_k: int = 5) -> list[float]:
        """
        Find the LLM scores for hypotheses with embeddings most similar to the given one.
        Used for tuning MR.Q to better align with high-quality LLM scores.
        """
        # Step 1: Embed the current hypothesis
        query_emb = self.memory.embedding.get_or_create(hypothesis_text)

        # Step 2: Search similar embeddings
        similar_items = self.memory.embedding.similarity_search(query_emb, top_k)

        llm_scores = []
        for item in similar_items:
            # Assumes item includes 'text' or 'id' to retrieve full score record
            matched_text = item.get("text")

            # Step 3: Query LLM score from database (or memory index)
            score_entry = self.memory.score.find_by_text_and_dimension(
                matched_text, dimension=dimension, source="llm"
            )
            if score_entry:
                llm_scores.append(score_entry.score)

        return llm_scores


    def align_with_llm_score(self, dimension: str, goal: dict, hypothesis: dict, llm_score: float):
        """
        Compare MR.Q score with LLM score and train the tuner.
        """
        mrq_score = self._estimate_score(goal, hypothesis, dimension)
        self.regression_tuners[dimension].add_example(mrq_score, llm_score)
        self.logger.log(
            "MRQAlignmentDataAdded",
            {
                "dimension": dimension,
                "mrq_score": mrq_score,
                "llm_score": llm_score,
            },
        )

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
