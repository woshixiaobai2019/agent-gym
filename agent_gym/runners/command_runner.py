"""Command Line Environment Runner."""

from typing import Dict, Any
from ..envs.CommandLineEnvironment import CommandLineEnvironment
from ..agents.miaruAgent import MirauAgent
from .base_runner import BaseRunner


class CommandLineRunner(BaseRunner):
    """Runner for Command Line Environment."""
    
    def __init__(self, data_file: str,
                 agent_base_url: str = "http://localhost:7996",
                 log_dir: str = "logs/command_line"):
        # Store command line specific parameters
        self.agent_base_url = agent_base_url
        
        # Call parent constructor
        super().__init__(data_file, log_dir)
    
    def _create_agent(self):
        return MirauAgent(base_url=self.agent_base_url)
    
    def _create_environment(self, task_id: int):
        return CommandLineEnvironment(data_file=self.data_file, task_id=task_id)
    
    def _get_runner_name(self) -> str:
        return "Command Line"
    
    def _get_metadata_extras(self) -> Dict[str, Any]:
        return {
            "agent_base_url": self.agent_base_url
        }
    
    def _print_task_details(self, task_info: Dict[str, Any]):
        query = task_info.get('query', 'N/A')
        
        # Display query
        if len(query) > 200:
            query = query[:200] + "..."
        print(f"Query: {query}")
        
        # Display environment setup if available
        if 'env' in task_info:
            env_info = task_info['env']
            if isinstance(env_info, dict):
                # If env is a dict, show key details
                for key, value in env_info.items():
                    if isinstance(value, str) and len(value) < 100:
                        print(f"  {key}: {value}")
            elif isinstance(env_info, str) and len(env_info) < 150:
                print(f"Environment Setup: {env_info}")
    
    def _print_initial_observation(self, observation: str):
        print(f"\n💻 QUERY: {observation}")
    
    def _print_environment_response(self, observation: str, reward: float, done: bool, total_reward: float):
        # Command line output can be long, so we might want to truncate for display
        display_obs = observation
        if len(observation) > 500:
            display_obs = observation[:500] + "\n... (truncated)"
        
        print(f"💻 RESULT: {display_obs}")
        print(f"📊 Reward: {reward}, Done: {done}, Total Reward: {total_reward}")