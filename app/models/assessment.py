from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Assessment:
    """One entry from the SHL product catalog."""
    entity_id: str
    name: str
    url: str
    description: str
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration_minutes: Optional[int] = None
    remote_testing: bool = False
    adaptive_irt: bool = False
    keys: List[str] = field(default_factory=list)   # e.g. ["Knowledge & Skills", "Simulations"]
    test_type: str = "K"                             # K / P / A / S / B

    # Derived text for embedding (built once at index time)
    embed_text: str = ""

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "job_levels": self.job_levels,
            "languages": self.languages,
            "duration_minutes": self.duration_minutes,
            "remote_testing": self.remote_testing,
            "adaptive_irt": self.adaptive_irt,
            "keys": self.keys,
            "test_type": self.test_type,
        }

    @property
    def display_duration(self) -> str:
        if self.duration_minutes is None:
            return "N/A"
        return f"{self.duration_minutes} min"