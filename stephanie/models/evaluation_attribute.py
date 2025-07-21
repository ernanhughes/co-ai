# stephanie/models/evaluation_attribute.py

from sqlalchemy import Column, Float, Integer, String, ForeignKey, JSON
from sqlalchemy.orm import relationship
from stephanie.models.base import Base


class EvaluationAttributeORM(Base):
    __tablename__ = "evaluation_attributes"

    id = Column(Integer, primary_key=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False)

    # Uniquely identify this entry
    dimension = Column(String, nullable=False)
    source = Column(String, nullable=False)  # e.g. 'mrq', 'sicql', 'ebt', 'llm', etc.

    # Rich metrics per source
    raw_score = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    uncertainty = Column(Float, nullable=True)
    advantage = Column(Float, nullable=True)
    pi_value = Column(Float, nullable=True)
    q_value = Column(Float, nullable=True)
    v_value = Column(Float, nullable=True)

    extra = Column(JSON, default={})

    evaluation = relationship("EvaluationORM", back_populates="attributes")

    def to_dict(self):
        return {
            "dimension": self.dimension,
            "source": self.source,
            "raw_score": self.raw_score,
            "energy": self.energy,
            "uncertainty": self.uncertainty,
            "advantage": self.advantage,
            "pi": self.pi_value,
            "q": self.q_value,
            "v": self.v_value,
            "extra": self.extra
        }
