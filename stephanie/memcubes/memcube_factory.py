# stephanie/memcubes/memcube_factory.py
from stephanie.scoring.scorable import Scorable
from stephanie.memcubes.memcube import MemCube, MemCubeType
from stephanie.scoring.scorable_factory import ScorableFactory
from datetime import datetime

from sqlalchemy.sql import text

class MemCubeFactory:
    @staticmethod
    def from_scordable(
        scorable: Scorable,
        version: str = "v1",
        source: str = "scorable_factory",
        model: str = "default"
    ) -> MemCube:
        """
        Convert Scorable to MemCube with enhanced metadata
        """
        return MemCube(
            scorable=scorable,
            version=version,
            source=source,
            model=model,
            priority=5,  # Default priority
            sensitivity="public",  # Default sensitivity
            extra_data={
                "type": scorable.target_type.value,
                "source_hash": hash(scorable.text),
                "pipeline_origin": source
            }
        )

    @staticmethod
    def from_dict(data: dict) -> MemCube:
        """Load MemCube from serialized data"""
        scorable = ScorableFactory.from_dict(data["scorable"])
        return MemCube(
            scorable=scorable,
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_modified=datetime.fromisoformat(data["last_modified"]),
            source=data["source"],
            model=data["model"],
            priority=data["priority"],
            sensitivity=data["sensitivity"],
            ttl=data.get("ttl"),
            usage_count=data.get("usage_count", 0),
            extra_data=data.get("metadata", {})
        )
    
    def _generate_version(self, scorable: Scorable) -> str:
        """Generate version based on content stability"""
        query = """
        SELECT version FROM memcubes
        WHERE scorable_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
        result = self.db.execute(text(query), [scorable.id]).fetchone()
        
        if not result:
            return "v1"
        
        # Increment version based on content change
        current_version = result.version
        content_hash = hash(scorable.text)
        
        if content_hash != self._get_content_hash(current_version):
            return f"v{int(current_version[1:]) + 1}"
        
        return current_version  # Same content, return current version