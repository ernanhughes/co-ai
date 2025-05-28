# models/goal.py
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from .base import Base


class GoalORM(Base):
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True)
    goal_text = Column(String, nullable=False)
    goal_type = Column(String)
    focus_area = Column(String)
    strategy = Column(String)
    llm_suggested_strategy = Column(String)
    source = Column(String, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)

    prompts = relationship("PromptORM", back_populates="goal")
    hypotheses = relationship("HypothesisORM", back_populates="goal")
    pipeline_runs = relationship("PipelineRunORM", back_populates="goal")
    scores = relationship("ScoreORM", back_populates="goal")
    lookaheads = relationship("LookaheadORM", back_populates="goal")
    reflection_deltas = relationship("ReflectionDeltaORM", back_populates="goal")
    ideas = relationship("IdeaORM", back_populates="goal")
    method_plans = relationship("MethodPlanORM", back_populates="goal")
    sharpening_predictions = relationship("SharpeningPredictionORM", back_populates="goal")