"""
Multi-Task Learning (MTL) gradient strategies for YOLOP models.

This package contains implementations of various MTL optimization methods
that manipulate gradients during training to mitigate task conflicts.
"""

from .base import MTLStrategy
from .original import OriginalStrategy
from .pcgrad import PCGradStrategy
from .cagrad import CAGradStrategy
from .gradnorm import GradNormStrategy

__all__ = [
    'MTLStrategy',
    'OriginalStrategy',
    'PCGradStrategy', 
    'CAGradStrategy',
    'GradNormStrategy',
]

# Strategy factory function
def create_mtl_strategy(strategy_name: str, model, optimizer, **kwargs):
    """
    Factory function to create MTL strategy instances.
    
    Args:
        strategy_name: Name of the strategy ('original', 'pcgrad', 'cagrad', 'gradnorm')
        model: The multi-task model
        optimizer: The optimizer
        **kwargs: Additional strategy-specific parameters
        
    Returns:
        MTLStrategy instance
    """
    strategies = {
        'original': OriginalStrategy,
        'pcgrad': PCGradStrategy,
        'cagrad': CAGradStrategy,
        'gradnorm': GradNormStrategy,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown MTL strategy: {strategy_name}. Available: {list(strategies.keys())}")
        
    return strategies[strategy_name](model, optimizer, **kwargs)