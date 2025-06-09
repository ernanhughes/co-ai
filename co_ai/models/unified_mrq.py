from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from datetime import datetime
from co_ai.models.base import Base

class UnifiedMRQModelORM(Base):
    __tablename__ = "unified_mrq_models"

    id = Column(Integer, primary_key=True)
    dimension = Column(String, nullable=False)
    model_path = Column(Text, nullable=False)
    trained_on = Column(DateTime, default=datetime.utcnow)
    pair_count = Column(Integer)
    trainer_version = Column(String)
    notes = Column(Text)
    context = Column(JSON)

    def __repr__(self):
        return f"<UnifiedMRQModelORM(dimension='{self.dimension}', trained_on={self.trained_on})>"
