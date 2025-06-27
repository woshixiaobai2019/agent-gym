"""Exception definitions for Agent Gym."""


class AgentGymError(Exception):
    """Base exception for all Agent Gym errors."""
    pass


class EnvironmentError(AgentGymError):
    """Raised when there's an error in environment execution."""
    pass


class AgentError(AgentGymError):
    """Raised when there's an error in agent execution."""
    pass


class UserModelError(AgentGymError):
    """Raised when there's an error in user model execution."""
    pass


class InvalidActionError(EnvironmentError):
    """Raised when an invalid action is provided to environment."""
    pass


class APIError(AgentError):
    """Raised when there's an error with API calls."""
    pass