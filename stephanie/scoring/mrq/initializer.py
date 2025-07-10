# stephanie/scoring/mrq/initializer.py
from .model import MRQModel
from .encoder import TextEncoder
from stephanie.evaluator.hypothesis_value_predictor import HypothesisValuePredictor
from stephanie.evaluator.mrq_trainer import MRQTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner

def initialize_dimension(self, dimension):
    if not self.value_predictor:
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)
    if not self.encoder:
        self.encoder = TextEncoder().to(self.device)

    self.regression_tuners[dimension] = RegressionTuner(
        dimension=dimension, logger=self.logger
    )
    self.trainers[dimension] = MRQTrainer(
        memory=self.memory, logger=self.logger,
        value_predictor=self.value_predictor,
        encoder=self.encoder, device=self.device
    )
    self.models[dimension] = MRQModel(self.encoder, self.value_predictor, device=self.device)
    self.min_score_by_dim[dimension] = 0.0
    self.max_score_by_dim[dimension] = 1.0
    self.logger.log("MRQModelInitializing", {"dimension": dimension})
