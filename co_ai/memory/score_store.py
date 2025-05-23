from typing import Optional, List, Dict
from datetime import datetime
from co_ai.models.score import Score
from co_ai.memory.base_store import BaseStore


class ScoreStore(BaseStore):
    def __init__(self, db, logger=None):
        super().__init__(db, logger)
        self.name = "score"
        self.table_name = "scores"

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "score"

    def insert_score(self, score_data: dict):
        """
        Inserts a new score into the scores table.
        
        :param score_data: Dictionary containing keys matching the table fields.
        """
        query = """
            INSERT INTO scores (
                goal_id,
                hypothesis_id,
                agent_name,
                model_name,
                evaluator_name,
                score_type,
                score,
                score_text,
                strategy,
                reasoning_strategy,
                rationale,
                reflection,
                review,
                meta_review,
                run_id,
                metadata,
                created_at
            ) VALUES (
                %(goal_id)s,
                %(hypothesis_id)s,
                %(agent_name)s,
                %(model_name)s,
                %(evaluator_name)s,
                %(score_type)s,
                %(score)s,
                %(score_text)s,
                %(strategy)s,
                %(reasoning_strategy)s,
                %(rationale)s,
                %(reflection)s,
                %(review)s,
                %(meta_review)s,
                %(run_id)s,
                %(metadata)s,
                %(created_at)s
            )
            RETURNING id;
        """

        try:
            with self.db.cursor() as cur:
                cur.execute(query, score_data)
                score_id = cur.fetchone()[0]
                self.db.commit()
                if self.logger:
                    self.logger.log("ScoreInserted", {
                        "score_id": score_id,
                        "goal_id": score_data.get("goal_id"),
                        "hypothesis_id": score_data.get("hypothesis_id"),
                        "score": score_data.get("score"),
                        "score_type": score_data.get("score_type"),
                        "evaluator": score_data.get("evaluator_name")
                    })
                return score_id
        except Exception as e:
            self.db.rollback()
            if self.logger:
                self.logger.log("ScoreInsertFailed", {"error": str(e)})
            raise

    def get_by_goal_id(self, goal_id: int) -> List[Dict]:
        """
        Returns all scores associated with a specific goal.
        """
        query = "SELECT * FROM scores WHERE goal_id = %s ORDER BY created_at DESC"
        return self._execute_and_fetch(query, (goal_id,))

    def get_by_hypothesis_id(self, hypothesis_id: int) -> List[Dict]:
        """
        Returns all scores associated with a specific hypothesis.
        """
        query = "SELECT * FROM scores WHERE hypothesis_id = %s ORDER BY created_at DESC"
        return self._execute_and_fetch(query, (hypothesis_id,))

    def get_by_evaluator(self, evaluator_name: str) -> List[Dict]:
        """
        Returns all scores produced by a specific evaluator (e.g., 'llm', 'mrq').
        """
        query = "SELECT * FROM scores WHERE evaluator_name = %s ORDER BY created_at DESC"
        return self._execute_and_fetch(query, (evaluator_name,))

    def get_by_strategy(self, strategy: str) -> List[Dict]:
        """
        Returns all scores generated using a specific reasoning strategy.
        """
        query = "SELECT * FROM scores WHERE strategy = %s ORDER BY created_at DESC"
        return self._execute_and_fetch(query, (strategy,))

    def get_all(self, limit: int = 100) -> List[Dict]:
        """
        Returns the most recent scores up to a limit.
        """
        query = f"SELECT * FROM scores ORDER BY created_at DESC LIMIT {limit}"
        return self._execute_and_fetch(query)

    def _execute_and_fetch(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Helper method to execute a query and fetch results as list of dicts.
        """
        try:
            with self.db.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                result = [dict(zip(columns, row)) for row in rows]
                return result
        except Exception as e:
            if self.logger:
                self.logger.log("ScoreFetchFailed", {"error": str(e)})
            return []