"""Iterative refinement engine"""

from typing import List, Dict
from collections import defaultdict
from knowledge_state import KnowledgeState
from operators import ImprovementOperator


class IterativeRefinementEngine:
    """Iteratively apply improvement operators"""
    
    def __init__(self, operators: List[ImprovementOperator], max_iterations: int = 100):
        self.operators = operators
        self.max_iterations = max_iterations
        self.history = []
        
    def refine(self, initial_state: KnowledgeState, verbose: bool = False) -> Dict:
        """
        Iteratively apply improvement operators until convergence or max iterations
        Returns refinement process metrics
        """
        state = initial_state
        metrics = {
            'iterations': [],
            'convergence_iteration': None,
            'total_applications': 0,
            'operators_applied': defaultdict(int)
        }
        
        for iteration in range(self.max_iterations):
            iteration_metrics = {
                'iteration': iteration,
                'num_claims': len(state.claims),
                'modifications': []
            }
            
            any_modified = False
            
            for operator in self.operators:
                old_claim_count = len(state.claims)
                new_state, modified = operator.apply(state)
                
                if not operator.verify_invariants(state, new_state):
                    if verbose:
                        print(f"Invariant violation in {operator.name} at iteration {iteration}")
                    continue
                
                if modified:
                    any_modified = True
                    metrics['operators_applied'][operator.name] += 1
                    iteration_metrics['modifications'].append({
                        'operator': operator.name,
                        'claims_before': old_claim_count,
                        'claims_after': len(new_state.claims)
                    })
                
                state = new_state
            
            metrics['iterations'].append(iteration_metrics)
            
            if not any_modified:
                metrics['convergence_iteration'] = iteration
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        metrics['total_applications'] = sum(metrics['operators_applied'].values())
        metrics['final_state'] = state
        
        return metrics
    
    def analyze_stability(self, metrics: Dict) -> Dict:
        """Analyze stability properties of the refinement process"""
        iterations = metrics['iterations']
        claim_counts = [it['num_claims'] for it in iterations]
        
        is_monotonic = all(claim_counts[i] >= claim_counts[i+1] 
                          for i in range(len(claim_counts)-1))
        
        convergence_speed = (metrics['convergence_iteration'] 
                           if metrics['convergence_iteration'] else len(iterations))
        
        changes_per_iteration = [len(it['modifications']) for it in iterations]
        
        return {
            'is_monotonic': is_monotonic,
            'converged': metrics['convergence_iteration'] is not None,
            'convergence_speed': convergence_speed,
            'total_modifications': sum(changes_per_iteration),
            'claim_reduction': claim_counts[0] - claim_counts[-1] if claim_counts else 0,
            'final_claim_count': claim_counts[-1] if claim_counts else 0
        }