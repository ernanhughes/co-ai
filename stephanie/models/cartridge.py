# models/cartridge.py

from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text, JSON
from sqlalchemy.orm import relationship

from stephanie.models.base import Base

class CartridgeORM(Base):
    __tablename__ = 'cartridges'

    id = Column(Integer, primary_key=True)

    # Association
    goal_id = Column(Integer, ForeignKey('goals.id'), nullable=True)  # Optional link to goal
    goal = relationship("GoalORM", back_populates="cartridges")

    # Source metadata
    source_type = Column(Text, nullable=False)     # e.g., 'document', 'hypothesis', 'response'
    source_uri = Column(Text)                      # Original file / API reference
    markdown_content = Column(Text, nullable=False)   # Where the rendered content lives
    embedding_id = Column(Integer, ForeignKey('embeddings.id'))  # <-- New embedding reference
   

    # Core content
    title = Column(Text)
    summary = Column(Text)
    sections = Column(JSON)                          # {"Intro": "...", "Conclusion": "..."}
    triples = Column(JSON)                           # e.g., [("LLMs", "can be fine-tuned with", "LoRA")]

    # Domains
    domain_tags = Column(JSON)                       # e.g., ["machine learning", "ethics"]

    created_at = Column(DateTime, default=datetime.utcnow)

    domains_rel = relationship("CartridgeDomainORM", back_populates="cartridge", cascade="all, delete-orphan")
