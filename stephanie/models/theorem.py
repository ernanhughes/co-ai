from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from datetime import datetime
from stephanie.models.base import Base

# Association table for many-to-many relationship
theorem_cartridges = Table(
    "theorem_cartridges",
    Base.metadata,
    Column("theorem_id", Integer, ForeignKey("theorems.id"), primary_key=True),
    Column("cartridge_id", Integer, ForeignKey("cartridges.id"), primary_key=True),
)

class TheoremORM(Base):
    __tablename__ = "theorems"
    
    id = Column(Integer, primary_key=True)
    statement = Column(Text, nullable=False)
    proof = Column(Text, nullable=True)
    embedding_id = Column(Integer, ForeignKey("embeddings.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship with Cartridges
    cartridges = relationship(
        "CartridgeORM",
        secondary=theorem_cartridges,
        back_populates="theorems"
    )

