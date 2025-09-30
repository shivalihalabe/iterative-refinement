"""Core data structures for representing structured knowledge"""

from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class Claim:
    """A single claim extracted from a paper"""
    id: str
    text: str
    evidence: List[str]
    section: str
    confidence: float
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class KnowledgeState:
    """Structured representation of paper content"""
    claims: Dict[str, Claim]
    relationships: Dict[str, List[str]]
    metadata: Dict[str, any]
    
    def to_dict(self):
        return {
            'claims': {k: asdict(v) for k, v in self.claims.items()},
            'relationships': self.relationships,
            'metadata': self.metadata
        }