"""Rule-based improvement operators"""

from typing import Tuple, Set
from ..knowledge_state import KnowledgeState, Claim
from .base import ImprovementOperator


class NormalizeEvidenceOperator(ImprovementOperator):
    """Standardize evidence formatting and remove duplicates"""
    
    def __init__(self):
        super().__init__("normalize_evidence")
        
    def apply(self, state: KnowledgeState) -> Tuple[KnowledgeState, bool]:
        new_state = KnowledgeState(
            claims={},
            relationships=state.relationships.copy(),
            metadata=state.metadata.copy()
        )
        
        modified = False
        
        for claim_id, claim in state.claims.items():
            unique_evidence = list(set(claim.evidence))
            unique_evidence.sort()
            
            if len(unique_evidence) != len(claim.evidence) or unique_evidence != claim.evidence:
                modified = True
            
            new_claim = Claim(
                id=claim.id,
                text=claim.text,
                evidence=unique_evidence,
                section=claim.section,
                confidence=claim.confidence
            )
            new_state.claims[claim_id] = new_claim
        
        self.applications += 1
        return new_state, modified


class MergeDuplicateClaimsOperator(ImprovementOperator):
    """Merge claims with overlapping evidence"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        super().__init__("merge_duplicates")
        self.similarity_threshold = similarity_threshold
        
    def apply(self, state: KnowledgeState) -> Tuple[KnowledgeState, bool]:
        new_state = KnowledgeState(
            claims=state.claims.copy(),
            relationships=state.relationships.copy(),
            metadata=state.metadata.copy()
        )
        
        modified = False
        claims_to_remove: Set[str] = set()
        
        claim_list = list(new_state.claims.values())
        for i, claim1 in enumerate(claim_list):
            if claim1.id in claims_to_remove:
                continue
                
            for claim2 in claim_list[i+1:]:
                if claim2.id in claims_to_remove:
                    continue
                    
                evidence1 = set(claim1.evidence)
                evidence2 = set(claim2.evidence)
                
                if not evidence1 or not evidence2:
                    continue
                    
                overlap = len(evidence1 & evidence2)
                similarity = overlap / min(len(evidence1), len(evidence2))
                
                if similarity >= self.similarity_threshold:
                    claim1.evidence = list(evidence1 | evidence2)
                    claim1.confidence = max(claim1.confidence, claim2.confidence)
                    
                    if 'merges' not in new_state.metadata:
                        new_state.metadata['merges'] = []
                    new_state.metadata['merges'].append({
                        'kept': claim1.id,
                        'removed': claim2.id,
                        'similarity': similarity
                    })
                    
                    claims_to_remove.add(claim2.id)
                    modified = True
        
        for claim_id in claims_to_remove:
            del new_state.claims[claim_id]
            if claim_id in new_state.relationships:
                del new_state.relationships[claim_id]
        
        self.applications += 1
        return new_state, modified


class RemoveWeakClaimsOperator(ImprovementOperator):
    """Remove claims with insufficient evidence"""
    
    def __init__(self, min_confidence: float = 0.3):
        super().__init__("remove_weak")
        self.min_confidence = min_confidence
        
    def apply(self, state: KnowledgeState) -> Tuple[KnowledgeState, bool]:
        new_state = KnowledgeState(
            claims={},
            relationships=state.relationships.copy(),
            metadata=state.metadata.copy()
        )
        
        modified = False
        removed_claims = []
        
        for claim_id, claim in state.claims.items():
            if claim.confidence >= self.min_confidence:
                new_state.claims[claim_id] = claim
            else:
                removed_claims.append(claim_id)
                modified = True
        
        for claim_id in removed_claims:
            if claim_id in new_state.relationships:
                del new_state.relationships[claim_id]
        
        if modified:
            if 'weak_removals' not in new_state.metadata:
                new_state.metadata['weak_removals'] = []
            new_state.metadata['weak_removals'].extend(removed_claims)
        
        self.applications += 1
        return new_state, modified