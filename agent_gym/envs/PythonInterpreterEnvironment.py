"""Python Interpreter Environment for Agent Gym."""

import json
import requests
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from math_verify import parse, verify
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from ..core.base import StaticEnvironment, timeout_context, TimeoutError
from ..core.types import ObservationType, ActionType, ToolsType, Reward, Done
from ..core.exceptions import EnvironmentError


class PythonInterpreterEnvironment(StaticEnvironment):
    """Python interpreter environment that provides code execution capabilities."""
    
    def __init__(self, data_file: str = "agent_gym/data/py.json", task_id: int = 0,
                 sandbox_url: str = "http://localhost:8080",
                 request_timeout: int = 30):
        """Initialize the Python interpreter environment.
        
        Args:
            data_file: Path to JSON file containing tasks
            task_id: Index of the task to load from data file
            sandbox_url: URL of the sandbox code execution service
            request_timeout: Request timeout in seconds
        """
        super().__init__(data_file, task_id)
        self.sandbox_url = sandbox_url.rstrip('/')
        self.request_timeout = request_timeout
        self.tools = self._define_tools()
        
        if data_file:
            self._load_task_data()
    
    def reset(self) -> Tuple[ObservationType, ToolsType]:
        """Reset the environment and return initial question."""
        if not self.current_task:
            raise EnvironmentError("No task loaded")
        
        question = self.current_task.get("question", "Please solve the problem using Python.")
        return question, self.tools
    
    def step(self, action: ActionType, timeout: int = 30) -> Tuple[ObservationType, Reward, Done]:
        """Execute an action with timeout."""
        try:
            with timeout_context(timeout):
                if self._is_tool_call(action):
                    observation = self._execute_tool_call(action)
                    reward = 0.0
                    done = False
                else:
                    # Agent provided final answer
                    observation = self._evaluate_final_answer(action)
                    reward = 1.0 if self._verify_task_completion(action) else 0.0
                    done = True
                
                return observation, reward, done
                
        except TimeoutError as e:
            return f"Timeout error: {str(e)}", -0.5, False
        except Exception as e:
            return f"Execution error: {str(e)}", -0.1, False
    
    def _is_tool_call(self, action: ActionType) -> bool:
        """Check if the action is a tool call."""
        if isinstance(action, dict):
            return "tool_calls" in action and action["tool_calls"]
        return False
    
    def _execute_tool_call(self, action: ActionType) -> str:
        """Execute a tool call and return the result."""
        try:
            # Parse tool call (assuming OpenAI format)
            tool_calls = action.get("tool_calls", [])
            results = []
            
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                if function_name == "run_python_code":
                    result = self._run_python_code(arguments["code"])
                    results.append(result)
                else:
                    results.append(f"Error: Unknown function '{function_name}'")
            
            return "\n\n".join(results)
            
        except Exception as e:
            return f"Tool execution error: {str(e)}"
    
    def _run_python_code(self, code: str) -> str:
        """Execute Python code using the sandbox API."""
        try:
            payload = {
                "code": code,
                "language": "python"
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.sandbox_url}/run_code",
                json=payload,
                headers=headers,
                timeout=self.request_timeout
            )
            
            if response.status_code != 200:
                return f"Sandbox API error: {response.status_code} - {response.text}"
            
            result = response.json()
            return self._format_execution_result(result)
            
        except requests.exceptions.Timeout:
            return "Code execution timeout"
        except requests.exceptions.RequestException as e:
            return f"Request error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def _format_execution_result(self, result: Dict[str, Any]) -> str:
        """Format the execution result for display."""
        status = result.get("status", "Unknown")
        
        if status == "Failed":
            # Handle compilation or runtime errors
            message = result.get("message", "")
            run_result = result.get("run_result", {})
            
            if run_result:
                stderr = run_result.get("stderr", "")
                stdout = run_result.get("stdout", "")
                execution_time = run_result.get("execution_time", 0)
                return_code = run_result.get("return_code", 0)
                
                output_parts = []
                if stdout:
                    output_parts.append(f"STDOUT:\n{stdout}")
                if stderr:
                    output_parts.append(f"STDERR:\n{stderr}")
                
                output_parts.append(f"Execution time: {execution_time:.4f}s")
                output_parts.append(f"Return code: {return_code}")
                
                return "\n\n".join(output_parts)
            else:
                return f"Execution failed: {message}"
        
        elif status == "Success":
            # Handle successful execution
            run_result = result.get("run_result", {})
            stdout = run_result.get("stdout", "")
            stderr = run_result.get("stderr", "")
            execution_time = run_result.get("execution_time", 0)
            
            output_parts = []
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                output_parts.append(f"STDERR:\n{stderr}")
            
            output_parts.append(f"Execution time: {execution_time:.4f}s")
            
            return "\n\n".join(output_parts)
        
        else:
            return f"Unknown status: {status}\nRaw result: {json.dumps(result, indent=2)}"
    
    def _evaluate_final_answer(self, action: ActionType) -> str:
        """Evaluate agent's final answer."""
        if isinstance(action, dict) and "content" in action:
            content = action["content"]
        else:
            content = str(action)
        
        return f"Agent's final answer: {content}"
    
    def _verify_task_completion(self, action: ActionType) -> bool:
        """Verify if the task has been completed successfully."""
        if not self.current_task or "answer" not in self.current_task:
            return False
        
        # Extract predicted answer from action
        if isinstance(action, dict) and "content" in action:
            predicted_answer = action["content"]
        else:
            predicted_answer = str(action)
        
        correct_answer = self.current_task["answer"]
        
        # Use the verify_answer function
        return self.verify_answer(correct_answer, predicted_answer)
    
    def verify_answer(self, extracted_answer: str, correct_answer: str) -> bool:
        """
        使用Math-Verify库比较数学答案
        """
        reward = 0
        if not extracted_answer:
            reward = 0
        else:
            # 处理反斜杠
            if extracted_answer.count("\\") % 2 == 1:
                # 如果提取的答案是单反斜杠，转换为双反斜杠
                extracted_answer = extracted_answer.replace("\\", "\\\\")
            
            # 解析标准答案
            gold_parsed = parse(correct_answer, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            
            if len(gold_parsed) != 0:
                # 解析提取的答案
                answer_parsed = parse(
                    extracted_answer,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=True,  # 允许数字
                                malformed_operators=True,  # 允许运算符
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=True,  # 允许在没有特定标记的情况下提取
                        )
                    ],
                    extraction_mode='first_match',
                )
                
                # 验证答案是否正确
                reward = float(verify(answer_parsed, gold_parsed))
        return bool(reward)

    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available tools in OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_python_code",
                    "description": "Execute Python code in a secure sandbox environment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ]
    
    def cleanup(self):
        """Clean up resources."""
        # No persistent resources to clean up
        pass