# stephanie\scoring\mrq\scorer.py
class MRQScorer:
    def __init__(self, mrq_model):
        self.model = mrq_model
        self.model.eval_mode()

    def score(self, goal: str, chunk: str) -> float:
        return self.model.predict(goal, chunk)
