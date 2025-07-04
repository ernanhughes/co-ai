import re
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_result import ScoreResult

class ProximityScorer(BaseScorer):

    def __init__(self, cfg, memory, logger, prompt_loader=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.prompt_loader = prompt_loader


    @property
    def name(self) -> str:
        return "proximity"

    def evaluate(self, prompt: str, response: str) -> ScoreBundle:
        """
        Evaluate the proximity between prompt and response using heuristic signals
        extracted from structured markdown blocks.

        Returns a ScoreBundle containing ScoreResult entries for each dimension.
        """
        if not response:
            return self._fallback("No proximity response available.")

        try:
            themes = self._extract_block(response, "Common Themes Identified")
            grafts = self._extract_block(response, "Grafting Opportunities")
            directions = self._extract_block(response, "Strategic Directions")

            # Simple scoring
            themes_score = 10.0 * len(themes)
            grafts_score = 10.0 * len(grafts)
            directions_score = 20.0 * len(directions)

            results = {
                "proximity_themes": ScoreResult(
                    dimension="proximity_themes",
                    score=min(100.0, themes_score),
                    weight=0.3,
                    rationale=f"{len(themes)} theme(s) identified",
                    source="proximity"
                ),
                "proximity_grafts": ScoreResult(
                    dimension="proximity_grafts",
                    score=min(100.0, grafts_score),
                    weight=0.3,
                    rationale=f"{len(grafts)} grafting suggestion(s)",
                    source="proximity"
                ),
                "proximity_directions": ScoreResult(
                    dimension="proximity_directions",
                    score=min(100.0, directions_score),
                    weight=0.4,
                    rationale=f"{len(directions)} strategic direction(s)",
                    source="proximity"
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

    def _generate_justification(self, themes, grafts, directions) -> str:
        return (
            f"Identified {len(themes)} themes, {len(grafts)} grafting suggestions, "
            f"and {len(directions)} strategic directions."
        )


    def _fallback(self, message: str) -> ScoreBundle:
        results = {
            "proximity_themes": ScoreResult(
                dimension="proximity_themes",
                score=0.0,
                weight=0.3,
                rationale=message,
                source="proximity",
            ),
            "proximity_grafts": ScoreResult(
                dimension="proximity_grafts",
                score=0.0,
                weight=0.3,
                rationale=message,
                source="proximity",
            ),
            "proximity_directions": ScoreResult(
                dimension="proximity_directions",
                score=0.0,
                weight=0.4,
                rationale=message,
                source="proximity",
            ),
        }
        return ScoreBundle(results=results)
    


    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        results = {}
        for dim in dimensions:
            vec = self._build_feature_vector(goal, scorable)

            # Dynamic training if needed
            if not self.trained[dim]:
                self._try_train_on_dimension(dim)

            if not self.trained[dim]:
                score = 50.0
                rationale = f"SVM not trained for {dim}, returning neutral."
            else:
                x = self.scalers[dim].transform([vec])
                raw_score = self.models[dim].predict(x)[0]
                tuned_score = self.regression_tuners[dim].transform(raw_score)
                score = tuned_score
                rationale = f"SVM predicted and aligned score for {dim}"

            self.logger.log("SVMScoreComputed", {
                "dimension": dim,
                "score": score,
                "hypothesis": scorable.text,
            })

            results[dim] = ScoreResult(
                dimension=dim,
                score=score,
                rationale=rationale,
                weight=1.0,
                source="svm",
            )

        return ScoreBundle(results=results)
