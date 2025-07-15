# stephanie/models/protocol_orm.py
from sqlalchemy import Column, String, JSONB
from stephanie.models.base import Base

class ProtocolORM(Base):
    __tablename__ = 'protocols'

    name = Column(String, primary_key=True)
    description = Column(String)
    input_format = Column(JSONB)
    output_format = Column(JSONB)
    failure_modes = Column(JSONB)
    depends_on = Column(JSONB)
    tags = Column(JSONB)
    capability = Column(String)
    preferred_for = Column(JSONB)
    avoid_for = Column(JSONB)