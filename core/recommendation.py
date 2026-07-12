"""
core/recommendation.py
───────────────────────
Structured recommendation dataclass passed through the pipeline.
Agents append Recommendation objects to state.recommendations;
humans review and record decisions via the UI or checkpoints.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Recommendation:
    title: str
    recommendation: str
    rationale: str
    confidence: float                    # 0.0 – 1.0
    risk: str                            # "low", "medium", "high"
    requires_human_approval: bool = True
    approver: Optional[str] = None
    decision: Optional[str] = None       # "approved", "modified", "rejected", "overridden"
    decision_notes: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "title":                   self.title,
            "recommendation":          self.recommendation,
            "rationale":               self.rationale,
            "confidence":              self.confidence,
            "risk":                    self.risk,
            "requires_human_approval": self.requires_human_approval,
            "approver":                self.approver,
            "decision":                self.decision,
            "decision_notes":          self.decision_notes,
            "timestamp":               self.timestamp,
        }
