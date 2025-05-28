from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.dialects.postgresql import ARRAY, REAL

from co_ai.models.base import Base


class MRQPreferencePairORM(Base):
    __tablename__ = "mrq_preference_pairs"

    id = Column(Integer, primary_key=True)

    goal = Column(Text, nullable=False)
    prompt = Column(Text, nullable=False)

    output_a = Column(Text, nullable=False)
    output_b = Column(Text, nullable=False)
    preferred = Column(Text, nullable=False)  # 'a' or 'b'

    fmt_a = Column(Text)  # e.g., direct, short_cot, code, long_cot
    fmt_b = Column(Text)
    
    features = Column(JSON)  # Optional: extra metadata
    run_id = Column(Text)
    source = Column(Text)  # e.g., arm_dataloader, user, agent
    created_at = Column(DateTime, default=datetime.utcnow)