# models/prompt.py
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from co_ai.models.base import Base


class PromptORM(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True)

    # Agent and prompt metadata
    agent_name = Column(String, nullable=False)
    prompt_key = Column(String, nullable=False)  # e.g., generation_goal_aligned.txt
    prompt_text = Column(Text, nullable=False)
    response_text = Column(Text)  # Optional — if storing model output too
    goal_id = Column(Integer, ForeignKey("goals.id"))
    source = Column(String)  # e.g., manual, dsp_refinement, feedback_injection
    strategy = Column(String)  # e.g., goal_aligned, out_of_the_box
    version = Column(Integer, default=1)
    is_current = Column(Boolean, default=False)
    extra_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

    goal = relationship("GoalORM", back_populates="prompts")
    hypotheses = relationship("HypothesisORM", back_populates="prompt")
