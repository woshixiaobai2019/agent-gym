"""NLP Environment that uses LLM to simulate user and environment interactions."""

import os
import json
import requests
import time
import threading
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from ..core.base import StaticEnvironment, timeout_context, TimeoutError
from ..core.types import ObservationType, ActionType, ToolsType, Reward, Done
from ..core.exceptions import EnvironmentError


class NLPEnvironment(StaticEnvironment):
    """NLP Environment that uses LLM to simulate user interactions."""
    
    def __init__(self, data_file: str = None, task_id: int = 0, 
                 env_llm_base_url: str = "http://localhost:8000",
                 env_llm_api_key: str = "dummy",
                 model: str = "gemini-2.5-flash-nothinking",
                 max_turns: int = 20,
                 stream: bool = True,
                 request_timeout: int = 60):
        """Initialize the NLP environment.
        
        Args:
            data_file: Path to JSON file containing tasks
            task_id: Index of the task to load from data file
            env_llm_base_url: Base URL for the environment LLM API
            env_llm_api_key: API key for the environment LLM
            max_turns: Maximum number of conversation turns
            stream: Whether to use streaming responses
            request_timeout: Request timeout in seconds
        """
        super().__init__(data_file, task_id)
        self.env_llm_base_url = env_llm_base_url.rstrip('/')
        self.env_llm_api_key = env_llm_api_key
        self.max_turns = max_turns
        self.stream = stream
        self.model = model
        self.request_timeout = request_timeout
        self.current_turn = 0
        self.history = []  # Conversation history from env perspective
        self.system_prompt = self._load_system_prompt()
        
        # Thread pool for async-like behavior without event loop issues
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        if data_file:
            self._load_task_data()
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        try:
            prompt_path = Path(__file__).parent.parent / "data" / "system_prompt.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise EnvironmentError(f"Failed to load system prompt: {e}")
    
    def reset(self) -> Tuple[ObservationType, ToolsType]:
        """Reset the environment and start new conversation."""
        self.current_turn = 0
        self.history = []
        
        if not self.current_task:
            raise EnvironmentError("No task loaded")
        
        # Get initial response from env-llm (user's opening)
        initial_query = self._build_query(None, "")
        initial_response = self._call_env_llm(initial_query)
        
        if initial_response["type"] == "nlp":
            observation = initial_response["content"]
            self.history.append(f"Environment: {observation}")
            return observation, self.current_task.get("tools", [])
        else:
            raise EnvironmentError("Expected initial NLP response from environment")
    
    def step(self, action: ActionType, timeout: int = 30) -> Tuple[ObservationType, Reward, Done]:
        """Execute an action by sending it to env-llm."""
        try:
            with timeout_context(timeout):
                self.current_turn += 1
                
                # Validate and prepare current agent input
                current_agent_input = self._format_agent_input(action)
                extra_info = self._validate_action(action)
                
                # Add agent action to history
                self.history.append(f"Agent: {current_agent_input}")
                
                # Build query for env-llm
                query = self._build_query(current_agent_input, extra_info)
                
                # Call env-llm
                response = self._call_env_llm(query)
                
                # Process response
                return self._process_env_response(response)
                
        except TimeoutError as e:
            return f"Timeout error: {str(e)}", -0.5, False
        except Exception as e:
            return f"Environment error: {str(e)}", -0.1, False
    
    def _call_env_llm(self, query: str, max_retries: int = 3) -> Dict[str, Any]:
        """Call the environment LLM with streaming support and retries."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": self.stream
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.env_llm_api_key}"
        }
        
        for attempt in range(max_retries):
            try:
                if self.stream:
                    return self._stream_request(payload, headers)
                else:
                    return self._non_stream_request(payload, headers)
                    
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise TimeoutError(f"Environment LLM request timeout after {max_retries} attempts")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise EnvironmentError(f"Environment LLM request failed: {e}")
                time.sleep(2 ** attempt)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise EnvironmentError(f"Unexpected error calling environment LLM: {e}")
                time.sleep(2 ** attempt)
    
    def _stream_request(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Handle streaming request."""
        url = f"{self.env_llm_base_url}/v1/chat/completions"
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.request_timeout,
                stream=True
            )
            
            if response.status_code != 200:
                raise EnvironmentError(f"Env-LLM API error: {response.status_code} - {response.text}")
            
            content_buffer = ""
            
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                    
                line = line.strip()
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        if 'choices' in chunk_data and chunk_data['choices']:
                            delta = chunk_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content_buffer += delta['content']
                                # Print streaming content in real-time for debugging
                                print(delta['content'], end='', flush=True)
                                
                    except json.JSONDecodeError:
                        continue
            
            print()  # New line after streaming
            return self._parse_env_output(content_buffer)
            
        finally:
            if 'response' in locals():
                response.close()
    
    def _non_stream_request(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Handle non-streaming request."""
        url = f"{self.env_llm_base_url}/v1/chat/completions"
        payload['stream'] = False
        
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.request_timeout
        )
        
        if response.status_code != 200:
            raise EnvironmentError(f"Env-LLM API error: {response.status_code} - {response.text}")
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        return self._parse_env_output(content)
    
    def _format_agent_input(self, action: ActionType) -> str:
        """Format agent action into string representation."""
        if isinstance(action, dict):
            if "tool_calls" in action and action["tool_calls"]:
                # Format tool calls
                tool_strs = []
                for tool_call in action["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    try:
                        func_args = json.loads(tool_call["function"]["arguments"])
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in func_args.items())
                        tool_strs.append(f"{func_name}({args_str})")
                    except json.JSONDecodeError:
                        tool_strs.append(f"{func_name}(invalid_args)")
                return f"Tool calls: {'; '.join(tool_strs)}"
            elif "content" in action:
                return f"Message: {action['content']}"
            else:
                return str(action)
        else:
            return str(action)
    
    def _validate_action(self, action: ActionType) -> str:
        """Validate agent action and return extra info."""
        extra_info_parts = []
        
        # Check if max turns exceeded
        if self.current_turn > self.max_turns:
            extra_info_parts.append(f"Warning: Maximum turns ({self.max_turns}) exceeded")
        
        # Validate action format
        if isinstance(action, dict):
            if "tool_calls" in action and action["tool_calls"]:
                # Validate tool calls
                for i, tool_call in enumerate(action["tool_calls"]):
                    try:
                        if "function" not in tool_call:
                            extra_info_parts.append(f"Error: Tool call {i} missing 'function' field")
                            continue
                        
                        func = tool_call["function"]
                        if "name" not in func:
                            extra_info_parts.append(f"Error: Tool call {i} missing function name")
                        
                        if "arguments" not in func:
                            extra_info_parts.append(f"Error: Tool call {i} missing arguments")
                        else:
                            # Try to parse arguments as JSON
                            try:
                                json.loads(func["arguments"])
                            except json.JSONDecodeError as e:
                                extra_info_parts.append(f"Error: Tool call {i} has invalid JSON arguments: {e}")
                    
                    except Exception as e:
                        extra_info_parts.append(f"Error: Tool call {i} validation failed: {e}")
        
        return "; ".join(extra_info_parts) if extra_info_parts else ""
    
    def _build_query(self, current_agent_input: Optional[str], extra_info: str) -> str:
        """Build XML query for env-llm."""
        from datetime import datetime
        
        task = self.current_task
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build history string
        history_str = "\n".join(self.history) if self.history else "No previous interactions"
        
        # Build tools string
        tools_str = json.dumps(task.get("tools", []), ensure_ascii=False, indent=2)
        
        query = f"""<query>
<environment_description>{task.get('environment_description', '')}</environment_description>
<environment_type>{task.get('environment_type', '')}</environment_type>
<tools>{tools_str}</tools>
<story_stages>{json.dumps(task.get('story_stages', []), ensure_ascii=False, indent=2)}</story_stages>
<history>{history_str}</history>
<user_persona>{task.get('user_persona', '')}</user_persona>
<current_agent_input>{current_agent_input or ''}</current_agent_input>
<extra_info>{extra_info}</extra_info>
</query>"""
        
        return query
    
    def _parse_env_output(self, content: str) -> Dict[str, Any]:
        """Parse environment LLM output."""
        import re
        
        # Extract content between <output> tags
        pattern = r'<output>\s*(\{.*?\})\s*</output>'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            # Fallback: try to find JSON-like content
            json_pattern = r'\{[^{}]*"type"[^{}]*\}'
            json_match = re.search(json_pattern, content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise EnvironmentError(f"Invalid env-llm response format: {content}")
        else:
            json_str = match.group(1)
        
        try:
            output_json = json.loads(json_str)
            
            # Validate required fields
            if "type" not in output_json:
                raise EnvironmentError("Missing 'type' field in env-llm response")
            
            if "content" not in output_json:
                raise EnvironmentError("Missing 'content' field in env-llm response")
            
            return output_json
            
        except json.JSONDecodeError as e:
            raise EnvironmentError(f"Invalid JSON in env-llm response: {e}\nContent: {content}")
    
    def _process_env_response(self, response: Dict[str, Any]) -> Tuple[ObservationType, Reward, Done]:
        """Process environment LLM response."""
        response_type = response["type"]
        content = f'{response["content"]}'
        
        if response_type == "nlp":
            # Natural language response from user/environment
            finished = response.get("finished", False)
            success = response.get("success", False)
            
            # Add to history
            self.history.append(f"Environment: {content}")
            
            # Calculate reward and done status
            if finished:
                reward = 1.0 if success else 0.0
                done = True
            else:
                reward = 0.0
                done = False
            
            return content, reward, done
            
        elif response_type == "tool_response":
            # Tool execution response - 直接返回内容，不添加特殊格式
            # Add to history
            self.history.append(f"Tool Result: {content}")
            
            # Tool responses don't end the conversation
            return content, 0.0, False
            
        else:
            raise EnvironmentError(f"Unknown response type: {response_type}")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()