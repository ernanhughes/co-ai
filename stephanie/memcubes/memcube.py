# stephanie/memcubes/memcube.py
from datetime import datetime
from enum import Enum

from stephanie.scoring.scorable import Scorable
from stephanie.utils.file_utils import hash_text


class MemCubeType(Enum):
    DOCUMENT = "document"
    PROMPT = "prompt"
    RESPONSE = "response"
    HYPOTHESIS = "hypothesis"
    SYMBOL = "symbol"
    THEOREM = "theorem"
    TRIPLE = "triple"
    CARTRIDGE = "cartridge"

class MemCube:
    def __init__(
        self,
        scorable: Scorable,
        version: str = "v1",
        created_at: datetime = None,
        last_modified: datetime = None,
        source: str = "user_input",
        model: str = "llama3",
        access_policy: dict = None,
        priority: int = 5,  # 1–10 scale
        sensitivity: str = "public",  # public, internal, confidential
        ttl: int = None,  # Time-to-live in days
        usage_count: int = 0,
        extra_data: dict = None
    ):
        """
        MemCube wraps Scorable with versioning, governance, and lifecycle extra_data
        """
        self.id = f"{hash_text(scorable.text)}_{version}"
        self.scorable = scorable
        self.version = version
        self.created_at = created_at or datetime.utcnow()
        self.last_modified = last_modified or self.created_at
        self.source = source
        self.model = model
        self.priority = priority
        self.sensitivity = sensitivity
        self.ttl = ttl
        self.usage_count = usage_count
        self.extra_data = extra_data or {}
        self._validate_sensitivity()

    def _validate_sensitivity(self):
        """Ensure sensitivity tags are valid"""
        valid_tags = ["public", "internal", "confidential", "restricted"]
        if self.sensitivity not in valid_tags:
            raise ValueError(f"Sensitivity must be one of {valid_tags}")

    def to_dict(self) -> dict:
        """Convert to serializable format"""
        return {
            "id": self.id,
            "scorable": self.scorable.to_dict(),
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "source": self.source,
            "model": self.model,
            "priority": self.priority,
            "sensitivity": self.sensitivity,
            "ttl": self.ttl,
            "usage_count": self.usage_count,
            "extra_data": self.extra_data
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemCube":
        """Reconstruct from serialized dict"""
        from stephanie.scoring.scorable_factory import ScorableFactory
        scorable = ScorableFactory.from_dict(data["scorable"])
        return cls(
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
            extra_data=data.get("extra_data", {})
        )

    def increment_usage(self):
        """Update usage count and last access time"""
        self.usage_count += 1
        self.last_modified = datetime.utcnow()
        return self.usage_count

    def has_expired(self) -> bool:
        """Check if MemCube has reached its TTL"""
        if not self.ttl:
            return False
        from datetime import timedelta
        return (datetime.utcnow() - self.created_at) > timedelta(days=self.ttl)

    def apply_governance(self, user: str, action: str) -> bool:
        """Check access policies before allowing operation"""
        policy = self.extra_data.get("access_policy", {})
        allowed_roles = policy.get(action, ["admin", "researcher"])
        user_role = self._get_user_role(user)
        return user_role in allowed_roles

    def _get_user_role(self, user: str) -> str:
        """Get user role from identity or database"""
        # Replace with real implementation
        return "researcher"