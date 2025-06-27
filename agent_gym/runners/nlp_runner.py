"""NLP Environment Runner."""

from typing import Dict, Any
from ..envs.NLPEnvironment import NLPEnvironment
from ..agents.miaruAgent import MirauAgent
from .base_runner import BaseRunner


class NLPRunner(BaseRunner):
    """Runner for NLP Environment."""
    
    def __init__(self, data_file: str,
                 agent_base_url: str = "http://localhost:7996",
                 env_llm_base_url: str = "http://localhost:8000", 
                 env_llm_api_key: str = "dummy",
                 max_turns: int = 20, 
                 stream: bool = True, 
                 request_timeout: int = 60,
                 log_dir: str = "logs/nlp"):
        # Store NLP-specific parameters
        self.agent_base_url = agent_base_url
        self.env_llm_base_url = env_llm_base_url
        self.env_llm_api_key = env_llm_api_key
        self.env_max_turns = max_turns
        self.stream = stream
        self.request_timeout = request_timeout
        
        # Call parent constructor
        super().__init__(data_file, log_dir)
    
    def _create_agent(self):
        return MirauAgent(base_url=self.agent_base_url)
    
    def _create_environment(self, task_id: int):
        return NLPEnvironment(
            data_file=self.data_file,
            task_id=task_id,
            env_llm_base_url=self.env_llm_base_url,
            env_llm_api_key=self.env_llm_api_key,
            max_turns=self.env_max_turns,
            stream=self.stream,
            request_timeout=self.request_timeout
        )
    
    def _get_runner_name(self) -> str:
        return "NLP Environment"
    
    def _get_metadata_extras(self) -> Dict[str, Any]:
        return {
            "agent_base_url": self.agent_base_url,
            "env_llm_url": self.env_llm_base_url,
            "stream": self.stream,
            "request_timeout": self.request_timeout
        }
    
    def _print_task_details(self, task_info: Dict[str, Any]):
        env_desc = task_info.get('environment_description', 'N/A')
        user_persona = task_info.get('user_persona', 'N/A')
        
        # Truncate long descriptions for display
        if len(env_desc) > 150:
            env_desc = env_desc[:150] + "..."
        if len(user_persona) > 150:
            user_persona = user_persona[:150] + "..."
            
        print(f"Environment: {env_desc}")
        print(f"User Persona: {user_persona}")
    
    def _print_initial_observation(self, observation: str):
        print(f"\n🎭 USER: {observation}")
    
    def _print_environment_response(self, observation: str, reward: float, done: bool, total_reward: float):
        print(f"🎭 USER: {observation}")
        print(f"📊 Reward: {reward}, Done: {done}, Total Reward: {total_reward}")