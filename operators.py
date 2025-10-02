"""Improvement operators for knowledge states"""

import json
import os
import anthropic
from typing import Tuple, Set
from knowledge_state import KnowledgeState, Claim


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
        
    def verify_invariants(self, old_state: KnowledgeState, new_state: KnowledgeState) -> bool:
        """
        Verify that invariants are preserved after operation
        Basic invariant: never lose claims without merging
        """
        if len(new_state.claims) < len(old_state.claims):
            if 'merges' not in new_state.metadata:
                return False
        return True


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


class LLMMergeDuplicateClaimsOperator(ImprovementOperator):
    """Use LLM to identify and merge semantically duplicate claims"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("llm_merge_duplicates")
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        
    def apply(self, state: KnowledgeState) -> Tuple[KnowledgeState, bool]:
        if len(state.claims) < 2:
            return state, False
        
        claims_list = [
            {'id': cid, 'text': c.text, 'evidence': c.evidence}
            for cid, c in state.claims.items()
        ]
        
        prompt = f"""Analyze these claims and identify semantic duplicates.

Claims:
{json.dumps(claims_list, indent=2)}

Return JSON:
{{
    "merges": [
        {{
            "keep": "c1",
            "remove": ["c2"],
            "reason": "explanation"
        }}
    ]
}}

If no merges needed: {{"merges": []}}
Return ONLY the JSON."""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        try:
            merge_data = json.loads(response_text)
        except json.JSONDecodeError:
            return state, False
        
        if not merge_data.get('merges'):
            return state, False
        
        new_state = KnowledgeState(
            claims=state.claims.copy(),
            relationships=state.relationships.copy(),
            metadata=state.metadata.copy()
        )
        
        modified = False
        claims_to_remove = set()
        
        for merge in merge_data['merges']:
            keep_id = merge['keep']
            remove_ids = merge['remove']
            
            if keep_id not in new_state.claims:
                continue
            
            kept_claim = new_state.claims[keep_id]
            evidence_set = set(kept_claim.evidence)
            
            for remove_id in remove_ids:
                if remove_id in new_state.claims:
                    removed_claim = new_state.claims[remove_id]
                    evidence_set.update(removed_claim.evidence)
                    claims_to_remove.add(remove_id)
                    modified = True
            
            if modified:
                kept_claim.evidence = list(evidence_set)
                
                if 'merges' not in new_state.metadata:
                    new_state.metadata['merges'] = []
                new_state.metadata['merges'].append({
                    'kept': keep_id,
                    'removed': list(remove_ids),
                    'reason': merge.get('reason', 'semantic similarity')
                })
        
        for claim_id in claims_to_remove:
            del new_state.claims[claim_id]
        
        self.applications += 1
        return new_state, modified


class LLMExtractImplicitAssumptionsOperator(ImprovementOperator):
    """Use LLM to identify and extract implicit assumptions"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("llm_extract_assumptions")
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        
    def apply(self, state: KnowledgeState) -> Tuple[KnowledgeState, bool]:
        if not state.claims:
            return state, False
        
        sample_size = min(3, len(state.claims))
        sample_claims = list(state.claims.values())[:sample_size]
        claims_text = "\n".join([f"- {c.text}" for c in sample_claims])
        
        prompt = f"""Identify implicit assumptions in these claims.

Claims:
{claims_text}

Return JSON:
{{
    "assumptions": [
        {{
            "text": "assumption",
            "related_claim_ids": ["c1"],
            "confidence": 0.8
        }}
    ]
}}

If none found: {{"assumptions": []}}
Return ONLY the JSON."""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        try:
            assumption_data = json.loads(response_text)
        except json.JSONDecodeError:
            return state, False
        
        if not assumption_data.get('assumptions'):
            return state, False
        
        new_state = KnowledgeState(
            claims=state.claims.copy(),
            relationships=state.relationships.copy(),
            metadata=state.metadata.copy()
        )
        
        modified = False
        next_id = len(new_state.claims) + 1
        
        for assumption in assumption_data['assumptions']:
            new_claim = Claim(
                id=f"a{next_id}",
                text=f"[ASSUMPTION] {assumption['text']}",
                evidence=["derived by LLM"],
                section="assumptions",
                confidence=assumption.get('confidence', 0.6)
            )
            new_state.claims[new_claim.id] = new_claim
            next_id += 1
            modified = True
            
            for related_id in assumption.get('related_claim_ids', []):
                if related_id in new_state.relationships:
                    new_state.relationships[related_id].append(new_claim.id)
                else:
                    new_state.relationships[related_id] = [new_claim.id]
        
        self.applications += 1
        return new_state, modified