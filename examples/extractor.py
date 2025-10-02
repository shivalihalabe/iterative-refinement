"""LLM-powered extraction of claims from unstructured text"""

import json
import os
from typing import Optional
import anthropic

from knowledge_state import KnowledgeState, Claim


class LLMExtractor:
    """Extract structured claims from unstructured text using Claude"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        
    def extract_claims(self, paper_text: str) -> KnowledgeState:
        """Extract claims from paper text into structured format"""
        
        prompt = f"""Extract the key claims from this research paper text. For each claim, provide:
1. A clear statement of the claim
2. Evidence supporting it (quote relevant phrases)
3. The section it appears in
4. A confidence score (0-1) based on how well-supported it is

Return as JSON:
{{
    "claims": [
        {{
            "id": "c1",
            "text": "claim statement",
            "evidence": ["quote 1", "quote 2"],
            "section": "introduction",
            "confidence": 0.9
        }}
    ]
}}

Paper text:
{paper_text}

Return ONLY the JSON."""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        # extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        data = json.loads(response_text)
        
        claims = {}
        for claim_data in data['claims']:
            claim = Claim(
                id=claim_data['id'],
                text=claim_data['text'],
                evidence=claim_data['evidence'],
                section=claim_data.get('section', 'unknown'),
                confidence=claim_data.get('confidence', 0.5)
            )
            claims[claim.id] = claim
        
        return KnowledgeState(
            claims=claims,
            relationships={},
            metadata={'source': 'llm_extraction', 'model': 'claude-sonnet-4'}
        )