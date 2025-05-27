# models/search_results.py
from sqlalchemy import ARRAY, Column, Integer, String, ForeignKey, JSON, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base


class SearchResultORM(Base):
    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    result_type = Column(String)
    title = Column(Text)
    summary = Column(Text)
    url = Column(Text)
    author = Column(String)
    published_at = Column(DateTime)
    tags = Column(ARRAY(String))
    goal_id = Column(Integer, ForeignKey("goals.id"))
    parent_goal = Column(Text)
    strategy = Column(String)
    focus_area = Column(String)
    extra_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self, include_relationships: bool = False) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "source": self.source,
            "result_type": self.result_type,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "author": self.author,
            "published_at": self.published_at.isoformat()
            if self.published_at
            else None,
            "tags": self.tags,
            "goal_id": self.goal_id,
            "parent_goal": self.parent_goal,
            "strategy": self.strategy,
            "focus_area": self.focus_area,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
