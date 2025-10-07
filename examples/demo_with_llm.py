"""Demo with LLM integration"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from extractor import LLMExtractor
from operators import (
    NormalizeEvidenceOperator,
    RemoveWeakClaimsOperator,
    LLMMergeDuplicateClaimsOperator,
    LLMExtractImplicitAssumptionsOperator
)
from engine import IterativeRefinementEngine


SAMPLE_TEXT = """
Abstract: We present a novel approach to text classification using transformer-based models.
Our experiments show that pre-trained language models achieve state-of-the-art results on
multiple benchmark datasets. We find that fine-tuning BERT on domain-specific data improves
performance by 15% compared to baseline methods.

Introduction: Deep learning has revolutionized natural language processing. Modern neural
networks require large amounts of training data to achieve good performance. Transformer
architectures, introduced by Vaswani et al., have become the dominant paradigm in NLP.

Methods: We use BERT, a transformer-based model, as our base architecture. The model is
pre-trained on a large corpus and then fine-tuned on our target task. Attention mechanisms
allow the model to focus on relevant parts of the input.

Results: Our approach achieves 94% accuracy on the test set, outperforming previous
state-of-the-art methods by 7%. We observe that larger models generally perform better,
though with diminishing returns beyond 300M parameters.
"""


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: Set ANTHROPIC_API_KEY environment variable")
        return
    
    print("Iterative Refinement with LLM Integration")
    print("=" * 60)
    
    # extract claims
    print("\nExtracting claims from paper text...")
    extractor = LLMExtractor()
    initial_state = extractor.extract_claims(SAMPLE_TEXT)
    print(f"Extracted {len(initial_state.claims)} claims")
    
    print("\nInitial claims:")
    for cid, claim in list(initial_state.claims.items())[:3]:
        print(f"  {cid}: {claim.text[:60]}...")
    
    # set up operators
    operators = [
        NormalizeEvidenceOperator(),
        LLMMergeDuplicateClaimsOperator(),
        LLMExtractImplicitAssumptionsOperator(),
        RemoveWeakClaimsOperator(min_confidence=0.5),
    ]
    
    # refine
    print(f"\nRunning iterative refinement...")
    engine = IterativeRefinementEngine(operators, max_iterations=20)