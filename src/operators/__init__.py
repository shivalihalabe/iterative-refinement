"""Improvement operators for knowledge states"""

from .base import ImprovementOperator
from .rule_based import (
    NormalizeEvidenceOperator,
    MergeDuplicateClaimsOperator,
    RemoveWeakClaimsOperator
)

__all__ = [
    'ImprovementOperator',
    'NormalizeEvidenceOperator',
    'MergeDuplicateClaimsOperator',
    'RemoveWeakClaimsOperator',
]