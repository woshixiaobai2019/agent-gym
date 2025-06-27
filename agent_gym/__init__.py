"""Function Calling Agent Gym - A standardized interface for training LLM agents."""

from .core import (
    Environment,
    Agent,
    ObservationType,
    ActionType, 
    Reward,
    Done
)

__version__ = "0.1.0"
__all__ = [
    'Environment',
    'Agent',
    'ObservationType', 
    'ActionType',
    'Reward',
    'Done'
]