
from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base
from stephanie.models.belief_cartridge import BeliefCartridgeORM


# stephanie/models/belief_cartridge.py
class BeliefCartridgeORM(Base):
    __tablename__ = "belief_cartridges"
    
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)
    idea = Column(Text)
    goal_id = Column(Integer, ForeignKey("goals.id"))
    domain = Column(String)
    efficiency_score = Column(Float)  # Composite efficiency metric
    efficiency_details = Column(JSON)  # Detailed efficiency metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    evaluations = relationship("EvaluationORM", back_populates="belief_cartridge")
    goal = relationship("GoalORM", back_populates="cartridges")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "idea": self.idea,
            "goal_id": self.goal_id,
            "domain": self.domain,
            "efficiency_score": self.efficiency_score,
            "efficiency_details": self.efficiency_details,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }