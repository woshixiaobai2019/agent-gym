"""Python Interpreter Environment Runner."""

from typing import Dict, Any
from ..envs.PythonInterpreterEnvironment import PythonInterpreterEnvironment
from ..agents.miaruAgent import MirauAgent
from .base_runner import BaseRunner


class PythonRunner(BaseRunner):
    """Runner for Python Interpreter Environment."""
    
    def __init__(self, data_file: str, 
                 agent_base_url: str = "http://localhost:7996",
                 sandbox_url: str = "http://localhost:8080", 
                 request_timeout: int = 30,
                 log_dir: str = "logs/python"):
        # Store agent-specific parameters
        self.agent_base_url = agent_base_url
        self.sandbox_url = sandbox_url
        self.request_timeout = request_timeout
        
        # Call parent constructor
        super().__init__(data_file, log_dir)
    
    def _create_agent(self):
        return MirauAgent(base_url=self.agent_base_url)
    
    def _create_environment(self, task_id: int):
        return PythonInterpreterEnvironment(
            data_file=self.data_file,
            task_id=task_id,
            sandbox_url=self.sandbox_url,
            request_timeout=self.request_timeout
        )
    
    def _get_runner_name(self) -> str:
        return "Python Interpreter"
    
    def _get_metadata_extras(self) -> Dict[str, Any]:
        return {
            "agent_base_url": self.agent_base_url,
            "sandbox_url": self.sandbox_url,
            "request_timeout": self.request_timeout
        }
    
    def _print_task_details(self, task_info: Dict[str, Any]):
        print(f"Question: {task_info.get('question', 'N/A')}")
        print(f"Expected Answer: {task_info.get('answer', 'N/A')}")
    
    def _format_action_for_display(self, action: Dict[str, Any]) -> str:
        if action.get("tool_calls"):
            code_blocks = []
            for tool_call in action["tool_calls"]:
                if tool_call["function"]["name"] == "run_python_code":
                    try:
                        import json
                        args = json.loads(tool_call["function"]["arguments"])
                        code = args.get("code", "")
                        code_blocks.append(f"```python\n{code}\n```")
                    except:
                        code_blocks.append("```python\n(invalid code)\n```")
            return f"🐍 Executing Python code:\n" + "\n\n".join(code_blocks)
        return super()._format_action_for_display(action)
    
    def _print_environment_response(self, observation: str, reward: float, done: bool, total_reward: float):
        if "STDOUT:" in observation or "STDERR:" in observation:
            print(f"🔧 EXECUTION RESULT:\n{observation}")
        else:
            print(f"📊 FINAL ANSWER: {observation}")
        print(f"📊 Reward: {reward}, Done: {done}, Total Reward: {total_reward}")