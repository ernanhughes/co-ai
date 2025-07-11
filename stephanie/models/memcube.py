# stephanie/models/memcube.py
from sqlalchemy import Column, Integer, Text, Boolean, JSON, ForeignKey, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class MemCubeORM(Base):
    __tablename__ = "memcubes"

    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey("goals.id", ondelete="CASCADE"))
    document_id = Column(Integer, ForeignKey("documents.id"))
    triplet_id = Column(Integer, ForeignKey("triplets.id"))

    scorable_type = Column(Text, nullable=False)  # e.g., "document", "triplet"
    current_text = Column(Text, nullable=False)
    refined_text = Column(Text)

    ebt_energy = Column(JSON)
    uncertainty_scores = Column(JSON)
    scores = Column(JSON)

    refinement_steps = Column(Integer)
    converged = Column(Boolean, default=False)
    used_llm_fallback = Column(Boolean, default=False)

    state = Column(Text, default="raw")  # raw, scored, refined, verified, invalid
    created_at = Column(TIMESTAMP, default=func.now())
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    def __repr__(self):
        return (
            f"<MemCubeORM(id={self.id}, goal_id={self.goal_id}, "
            f"scorable_type='{self.scorable_type}', state='{self.state}', "
            f"refinement_steps={self.refinement_steps}, "
            f"converged={self.converged})>"
        )
    def to_dict(self):
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "document_id": self.document_id,
            "triplet_id": self.triplet_id,
            "scorable_type": self.scorable_type,
            "current_text": self.current_text,
            "refined_text": self.refined_text,
            "ebt_energy": self.ebt_energy,
            "uncertainty_scores": self.uncertainty_scores,
            "scores": self.scores,
            "refinement_steps": self.refinement_steps,
            "converged": self.converged,
            "used_llm_fallback": self.used_llm_fallback,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    