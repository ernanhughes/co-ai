# models/score.py
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base

class ScoreORM(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey("goals.id"))
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"))
    agent_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    evaluator_name = Column(String, nullable=False)
    score_type = Column(String, nullable=False)
    score = Column(Float)
    score_text = Column(Text)
    strategy = Column(String)
    reasoning_strategy = Column(String)
    rationale = Column(Text)
    reflection = Column(Text)
    review = Column(Text)
    meta_review = Column(Text)
    run_id = Column(String)
    extra_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    goal = relationship("GoalORM", back_populates="scores")
    hypothesis = relationship("HypothesisORM", back_populates="scores")