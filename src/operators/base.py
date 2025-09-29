"""Base class for improvement operators"""

from typing import Tuple
from ..knowledge_state import KnowledgeState


class ImprovementOperator:
    """Base class for operators that improve knowledge states"""
    
    def __init__(self, name: str):
        self.name = name
        self.applications = 0
        
    def apply(self, state: KnowledgeState) -> Tuple[KnowledgeState, bool]:
        """
        Apply operator to state
        
        Returns (new_state, was_modified)
        """
        raise NotImplementedError
        
    def verify_invariants(self, old_state: KnowledgeState, 
                         new_state: KnowledgeState) -> bool:
        """
        Verify that invariants are preserved after operation
        
        Basic invariant: never lose claims without merging
        """
        if len(new_state.claims) < len(old_state.claims):
            if 'merges' not in new_state.metadata:
                return False
        return True