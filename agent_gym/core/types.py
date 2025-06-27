"""Type definitions for Agent Gym."""

from typing import Any, TypeVar, List, Dict

# Generic types for maximum flexibility
Observation = TypeVar('Observation')
Action = TypeVar('Action') 
Tools = TypeVar('Tools')  # 工具定义，可以是任意格式
Reward = float
Done = bool

# For type hints in implementations
ObservationType = Any
ActionType = Any
ToolsType = Any  # 支持OpenAI格式、自定义格式等