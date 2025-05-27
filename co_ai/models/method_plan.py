# models/method_plan.py
from sqlalchemy import Boolean, Column, Integer, String, ForeignKey, JSON, DateTime, Float
from sqlalchemy.orm import relationship
from .base import Base
from datetime import datetime


class MethodPlanORM(Base):
    __tablename__ = "method_plans"

    id = Column(Integer, primary_key=True)

    # Core Idea & Goal Linkage
    idea_text = Column(String, nullable=False)
    idea_id = Column(Integer, ForeignKey("ideas.id"), nullable=True)
    goal_id = Column(Integer, ForeignKey("goals.id"))

    # Research Design Fields
    research_objective = Column(String, nullable=False)
    key_components = Column(JSON)  # List of techniques/components
    experimental_plan = Column(String)  # Detailed steps for testing
    hypothesis_mapping = Column(String)  # Or JSON if you parse it
    search_strategy = Column(String)  # Keywords or tools to use
    knowledge_gaps = Column(String)
    next_steps = Column(String)

    # Supporting Info
    task_description = Column(String)
    baseline_method = Column(String)
    literature_summary = Column(String)
    code_plan = Column(String)  # Optional pseudocode
    focus_area = Column(String)
    strategy = Column(String)
    score_novelty = Column(Float)
    score_feasibility = Column(Float)
    score_impact = Column(Float)
    score_alignment = Column(Float)
    evolution_level = Column(Integer, default=0)
    parent_plan_id = Column(Integer, ForeignKey("method_plans.id"), nullable=True)
    is_refinement = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    goal = relationship("GoalORM", back_populates="method_plans")
    parent_plan = relationship("MethodPlanORM", remote_side=[id], backref="refinements")


    def to_dict(self, include_relationships: bool = False) -> dict:
        result = {
            "id": self.id,
            "idea_text": self.idea_text,
            "idea_id": self.idea_id,
            "research_objective": self.research_objective,
            "key_components": self.key_components,
            "experimental_plan": self.experimental_plan,
            "hypothesis_mapping": self.hypothesis_mapping,
            "search_strategy": self.search_strategy,
            "knowledge_gaps": self.knowledge_gaps,
            "next_steps": self.next_steps,
            "baseline_method": self.baseline_method,
            "literature_summary": self.literature_summary,
            "code_plan": self.code_plan,
            "focus_area": self.focus_area,
            "strategy": self.strategy,
            "score_novelty": self.score_novelty,
            "score_feasibility": self.score_feasibility,
            "score_impact": self.score_impact,
            "score_alignment": self.score_alignment,
            "evolution_level": self.evolution_level,
            "parent_plan_id": self.parent_plan_id,
            "is_refinement": self.is_refinement,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

        if include_relationships:
            result["goal"] = self.goal.to_dict() if self.goal else None
            result["parent_plan"] = self.parent_plan.to_dict() if self.parent_plan else None
            result["refinements"] = [r.to_dict() for r in self.refinements] if self.refinements else []

        return result