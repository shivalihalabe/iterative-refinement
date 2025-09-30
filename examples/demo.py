"""Demo script showing basic usage"""

import sys
sys.path.insert(0, '..')

from knowledge_state import KnowledgeState, Claim
from operators import (
    NormalizeEvidenceOperator,
    MergeDuplicateClaimsOperator,
    RemoveWeakClaimsOperator
)
from engine import IterativeRefinementEngine


def create_sample_state() -> KnowledgeState:
    """Create a sample knowledge state with some redundancy"""
    claims = {
        'c1': Claim('c1', 'Deep learning models require large datasets', 
                   ['paper1:fig2', 'paper1:table1'], 'introduction', 0.9),
        'c2': Claim('c2', 'Neural networks need substantial training data',
                   ['paper1:fig2', 'paper2:sec3'], 'background', 0.85),
        'c3': Claim('c3', 'Transformers achieve state-of-the-art results',
                   ['paper3:table2'], 'results', 0.95),
        'c4': Claim('c4', 'Attention mechanisms are important',
                   ['paper3:fig1'], 'methods', 0.4),
        'c5': Claim('c5', 'BERT uses transformer architecture',
                   ['paper4:sec2', 'paper4:sec2'], 'methods', 0.9),
    }
    
    return KnowledgeState(
        claims=claims,
        relationships={'c1': ['c2'], 'c3': ['c5']},
        metadata={'source': 'sample_paper.txt'}
    )


def main():
    print("Iterative Paper Refinement Demo")
    print("=" * 60)
    
    initial_state = create_sample_state()
    print(f"\nInitial state: {len(initial_state.claims)} claims")
    
    operators = [
        NormalizeEvidenceOperator(),
        MergeDuplicateClaimsOperator(similarity_threshold=0.5),
        RemoveWeakClaimsOperator(min_confidence=0.5),
    ]
    
    engine = IterativeRefinementEngine(operators, max_iterations=50)
    metrics = engine.refine(initial_state, verbose=True)
    
    stability = engine.analyze_stability(metrics)
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"Converged: {stability['converged']}")
    print(f"Convergence speed: {stability['convergence_speed']} iterations")
    print(f"Final claims: {stability['final_claim_count']}")
    
    final_state = metrics['final_state']
    print("\nFinal claims:")
    for claim_id, claim in final_state.claims.items():
        print(f"  {claim_id}: {claim.text[:60]}...")


if __name__ == "__main__":
    main()