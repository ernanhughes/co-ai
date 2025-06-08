# stores/score_store.py
import json
from typing import Optional

from sqlalchemy.orm import Session

from co_ai.models import RuleApplicationORM
from co_ai.models.goal import GoalORM
from co_ai.models.score import ScoreORM
from co_ai.models.score_rule_link import ScoreRuleLinkORM


class ScoreStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "scores"
        self.table_name = "scores"

    def insert(self, score: ScoreORM):
        """
        Inserts a new score into the database.
        Accepts a dictionary (e.g., from Score dataclass).
        """
        try:
            self.session.add(score)
            self.session.flush()  # To get ID immediately

            if self.logger:
                self.logger.log(
                    "ScoreStored",
                    {
                        "score_id": score.id,
                        "goal_id": score.goal_id,
                        "hypothesis_id": score.hypothesis_id,
                        "agent": score.agent_name,
                        "model": score.model_name,
                        "scores": score.scores,
                        "timestamp": score.created_at.isoformat(),
                    },
                )

            # Link score to rule application if possible
            if score.pipeline_run_id and score.goal_id:
                rule_apps = (
                    self.session.query(RuleApplicationORM)
                    .filter_by(pipeline_run_id=score.pipeline_run_id, goal_id=score.goal_id)
                    .all()
                )
                for ra in rule_apps:
                    link = ScoreRuleLinkORM(score_id=score.id, rule_application_id=ra.id)
                    self.session.add(link)
                self.logger.log(
                    "ScoreLinkedToRuleApplications",
                    {
                        "score_id": score.id,
                        "linked_rule_application_ids": [ra.id for ra in rule_apps],
                    },
                )

            self.session.refresh(score)
            self.session.commit()
            return score.id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("ScoreInsertFailed", {"error": str(e)})
            raise

    def get_by_goal_id(self, goal_id: int) -> list[dict]:
        """Returns all scores associated with a specific goal."""
        results = self.session.query(ScoreORM).join(GoalORM).filter(GoalORM.id == goal_id).all()
        return [self._orm_to_dict(r) for r in results]

    def get_by_goal_type(self, goal_type: str) -> list[dict]:
        """Returns all scores associated with a specific goal."""
        results = self.session.query(ScoreORM).join(GoalORM).filter(GoalORM.goal_type == goal_type).all()
        return [self._orm_to_dict(r) for r in results]


    def get_by_hypothesis_id(
        self,
        hypothesis_id: int,
        source: Optional[str] = None
    ) -> list[dict]:
        """Returns all scores associated with a specific hypothesis, optionally filtered by evaluator source."""
        query = self.session.query(ScoreORM).filter(ScoreORM.hypothesis_id == hypothesis_id)
        
        if source:
            query = query.filter(ScoreORM.evaluator_name == source)

        results = query.all()
        return [self._orm_to_dict(r) for r in results]

    def get_by_run_id(self, run_id: str) -> list[dict]:
        """Returns all scores associated with a specific pipeline run."""
        results = self.session.query(ScoreORM).filter(ScoreORM.run_id == run_id).all()
        return [self._orm_to_dict(r) for r in results]

    def get_by_pipeline_run_id(self, pipeline_run_id: int) -> list[dict]:
        """Returns all scores associated with a specific pipeline run."""
        results = self.session.query(ScoreORM).filter(ScoreORM.pipeline_run_id == pipeline_run_id).all()
        return [self._orm_to_dict(r) for r in results]


    def get_by_evaluator(self, evaluator_name: str) -> list[dict]:
        """Returns all scores produced by a specific evaluator (LLM, MRQ, etc.)"""
        results = self.session.query(ScoreORM).filter(ScoreORM.evaluator_name == evaluator_name).all()
        return [self._orm_to_dict(r) for r in results]

    def get_by_strategy(self, strategy: str) -> list[dict]:
        """Returns all scores generated using a specific reasoning strategy."""
        results = self.session.query(ScoreORM).filter(ScoreORM.strategy == strategy).all()
        return [self._orm_to_dict(r) for r in results]

    def get_all(self, limit: int = 100) -> list[dict]:
        """Returns the most recent scores up to a limit."""
        results = self.session.query(ScoreORM).order_by(ScoreORM.created_at.desc()).limit(limit).all()
        return [self._orm_to_dict(r) for r in results]

    def _orm_to_dict(self, row: ScoreORM) -> dict:
        """Converts an ORM object back to a dictionary format"""
        return {
            "id": row.id,
            "goal_id": row.goal_id,
            "hypothesis_id": row.hypothesis_id,
            "agent_name": row.agent_name,
            "model_name": row.model_name,
            "evaluator_name": row.evaluator_name,
            "scores": (
                row.scores if isinstance(row.scores, dict) else json.loads(row.scores)
            ) if row.scores else {},
            "strategy": row.strategy,
            "reasoning_strategy": row.reasoning_strategy,
            "pipeline_run_id": row.pipeline_run_id,
            "extra_data": (
                row.extra_data if isinstance(row.extra_data, dict) else json.loads(row.extra_data)
            ) if row.extra_data else {},
            "created_at": row.created_at,
        }
    
    def get_rules_for_score(self, score_id: int) -> list[int]:
        links = (
            self.session.query(ScoreRuleLinkORM.rule_application_id)
            .filter_by(score_id=score_id)
            .all()
        )
        return [rid for (rid,) in links]