# co_ai/scoring/svm_scorer.py

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from co_ai.scoring.base_scorer import BaseScorer
from co_ai.scoring.score_result import ScoreResult
from co_ai.scoring.score_bundle import ScoreBundle
from collections import defaultdict


class SVMScorer(BaseScorer):
    def __init__(self, cfg: dict, memory, logger, dimensions=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.dimensions = dimensions or ["alignment"]
        self.models = {dim: SVR() for dim in self.dimensions}
        self.scalers = {dim: StandardScaler() for dim in self.dimensions}
        self.trained = {dim: False for dim in self.dimensions}

    def train(self, samples_by_dim: dict[str, list[dict]]):
        """
        Train per-dimension SVM from labeled LLM/MRQ training data
        """
        for dim, samples in samples_by_dim.items():
            x = []
            y = []
            for sample in samples:
                prompt = sample["prompt"]
                hyp = sample["output"]
                score = sample["value"]
                feat = self._build_feature_vector({"goal_text": prompt}, {"text": hyp})
                x.append(feat)
                y.append(score)

            x = np.array(x)
            y = np.array(y)
            self.scalers[dim].fit(x)
            x_scaled = self.scalers[dim].transform(x)

            self.models[dim].fit(x_scaled, y)
            self.trained[dim] = True

            self.logger.log("SVMTrainingComplete", {
                "dimension": dim,
                "samples": len(samples),
                "score_min": float(np.min(y)),
                "score_max": float(np.max(y)),
            })

    def _build_feature_vector(self, goal: dict, hypothesis: dict):
        """
        Basic feature vector: concat prompt + hypothesis embeddings + MRQ raw score (if available)
        """
        emb_goal = self.memory.embedding.get_or_create(goal["goal_text"])
        emb_hyp = self.memory.embedding.get_or_create(hypothesis["text"])
        vec = emb_goal + emb_hyp

        # Optional MRQ bridge feature
        mrq = self.memory.score.find_by_text_and_dimension(
            hypothesis["text"], dimension="alignment", source="mrq"
        )
        if mrq:
            vec.append(mrq.score / 100.0)  # normalized to [0,1]
        else:
            vec.append(0.5)  # neutral if no MRQ score

        return vec

    def train_from_database(self):
        pair_samples = self.memory.mrq.get_training_pairs_by_dimension()
        samples_by_dim = self.convert_mrq_pairs_to_supervised_examples(pair_samples)

        for dim, examples in samples_by_dim.items():
            self.train_for_dimension(dim, examples)


    def convert_mrq_pairs_to_supervised_examples(self, pair_samples: list[dict]) -> dict[str, list[dict]]:
        """
        Converts MRQ-style contrastive training pairs into a flat list of (prompt, output, value)
        entries per dimension, suitable for supervised regression training.
        """
        per_dimension = defaultdict(list)
        for pair in pair_samples:
            dim = pair.get("dimension", "default")

            for side in ["a", "b"]:
                output = pair.get(f"output_{side}")
                score = pair.get(f"value_{side}")
                if output is not None and score is not None:
                    per_dimension[dim].append({
                        "prompt": pair["prompt"],
                        "output": output,
                        "value": score
                    })

        self.logger.log("SVMConvertedMRQPacks", {
            "dimensions": list(per_dimension.keys()),
            "total_samples": sum(len(v) for v in per_dimension.values())
        })

        return per_dimension

    def train_for_dimension(self, dimension: str, examples: list[dict]):
        X = []
        y = []
        for ex in examples:
            prompt_vec = self.memory.embedding.get_or_create(ex["prompt"])
            output_vec = self.memory.embedding.get_or_create(ex["output"])
            pair_vec = np.array(prompt_vec + output_vec)
            X.append(pair_vec)
            y.append(ex["value"])

        X = np.array(X)
        y = np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = SVR(kernel="linear")  # you can adjust kernel if needed
        model.fit(X_scaled, y)

        self.models[dimension] = (scaler, model)

        self.logger.log("SVMModelTrained", {
            "dimension": dimension,
            "num_samples": len(y)
        })

    def score(self, goal: dict, hypothesis: dict, dimensions: list[str]) -> ScoreBundle:
        results = {}
        prompt_vec = self.memory.embedding.get_or_create(goal.get("goal_text"))
        response_vec = self.memory.embedding.get_or_create(hypothesis.get("text"))
        input_vec = np.array(prompt_vec + response_vec).reshape(1, -1)

        for dim in dimensions:
            scaler, model = self.models.get(dim, (None, None))
            if scaler is None or model is None:
                self.logger.log("SVMModelMissing", {"dimension": dim})
                continue

            vec_scaled = scaler.transform(input_vec)
            score = float(model.predict(vec_scaled)[0])
            score = max(0.0, min(100.0, score)) # clip score to [0, 100]
            self.logger.log("SVMScoreComputed", {
                "dimension": dim,
                "score": score,
                "input_vector": input_vec.tolist()
            })
            results[dim] = ScoreResult(
                dimension=dim,
                score=score,
                rationale=f"SVM-regressed score for {dim}",
                source="svm"
            )

        return ScoreBundle(results)


    def _train_dimension(self, dim: str):
        pairs_by_dim = self.memory.mrq.get_training_pairs_by_dimension()
        samples = pairs_by_dim.get(dim, [])
        if not samples:
            self.logger.log("SVMNoTrainingData", {"dimension": dim})
            self.trained[dim] = False
            return

        X = []
        y = []
        for sample in samples:
            goal = {"goal_text": sample["prompt"]}
            for side in ["a", "b"]:
                hyp = {"text": sample[f"output_{side}"]}
                label = sample.get(f"value_{side}")
                if label is not None:
                    vec = self._build_feature_vector(goal, hyp)
                    X.append(vec)
                    y.append(label)

        if len(X) < 5:
            self.logger.log("SVMInsufficientTrainingData", {"dimension": dim, "count": len(X)})
            self.trained[dim] = False
            return

        X_scaled = self.scalers[dim].fit_transform(X)
        self.models[dim].fit(X_scaled, y)
        self.trained[dim] = True
        self.logger.log("SVMTrained", {"dimension": dim, "samples": len(X)})
