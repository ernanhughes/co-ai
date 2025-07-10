# stephanie/scoring/mrq/scorer.py

from .core_scoring import MRQCoreScoring
from .training import MRQTraining
from .model_io import MRQModelIO
from .initializer import initialize_dimension

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
        self.estimate_score = lambda goal, scorable, dim: estimate_score(self, goal, scorable, dim)
        self.evaluate = lambda prompt, response: evaluate(self, prompt, response)
        self.judge = lambda goal, prompt, a, b: judge(self, goal, prompt, a, b)
        self.predict_score_from_prompt = lambda prompt, dim="mrq", top_k=5: predict_score_from_prompt(self, prompt, dim, top_k)

        self.train_from_database = lambda cfg: train_from_database(self, cfg)
        self.train_from_context = lambda ctx, cfg: train_from_context(self, ctx, cfg)
        self.align_mrq_with_llm_scores_from_pairs = lambda samples, dim, prefix="MRQAlignment": align_mrq_with_llm_scores_from_pairs(self, samples, dim, prefix)
        self.update_score_bounds_from_data = lambda samples, dim: update_score_bounds_from_data(self, samples, dim)

        self.save_models = lambda: save_models(self)
        self.load_models = lambda: load_models(self)
        self.load_models_with_path = lambda: load_models_with_path(self)
        self.save_metadata = lambda base_dir: save_metadata(self, base_dir)
