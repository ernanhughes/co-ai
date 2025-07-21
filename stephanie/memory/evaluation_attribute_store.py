# stephanie/memory/evaluation_attribute_store.py

from typing import Optional, List
from sqlalchemy.orm import Session
from stephanie.models.evaluation_attribute import EvaluationAttributeORM

class EvaluationAttributeStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "evaluation_attributes"
        self.table_name = "evaluation_attributes"

    def insert(self, attribute: EvaluationAttributeORM) -> int:
        try:
            self.session.add(attribute)
            self.session.commit()

            if self.logger:
                self.logger.log("AttributeStored", {
                    "evaluation_id": attribute.evaluation_id,
                    "dimension": attribute.dimension,
                    "source": attribute.source,
                    "score": attribute.score,
                    "energy": attribute.energy,
                    "uncertainty": attribute.uncertainty,
                    "weight": attribute.weight
                })

            return attribute.id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("AttributeInsertFailed", {"error": str(e)})
            raise

    def get_by_evaluation_id(self, evaluation_id: int) -> List[EvaluationAttributeORM]:
        return (
            self.session.query(EvaluationAttributeORM)
            .filter_by(evaluation_id=evaluation_id)
            .all()
        )

    def get_by_dimension(self, evaluation_id: int, dimension: str) -> List[EvaluationAttributeORM]:
        return (
            self.session.query(EvaluationAttributeORM)
            .filter_by(evaluation_id=evaluation_id, dimension=dimension)
            .all()
        )

    def get_by_source(self, evaluation_id: int, dimension: str, source: str) -> Optional[EvaluationAttributeORM]:
        return (
            self.session.query(EvaluationAttributeORM)
            .filter_by(evaluation_id=evaluation_id, dimension=dimension, source=source)
            .first()
        )

    def delete_by_evaluation(self, evaluation_id: int) -> int:
        deleted = (
            self.session.query(EvaluationAttributeORM)
            .filter_by(evaluation_id=evaluation_id)
            .delete()
        )
        self.session.commit()
        return deleted

    def get_all_for_dimension(self, dimension: str, limit: int = 100) -> List[EvaluationAttributeORM]:
        return (
            self.session.query(EvaluationAttributeORM)
            .filter_by(dimension=dimension)
            .order_by(EvaluationAttributeORM.created_at.desc())
            .limit(limit)
            .all()
        )
