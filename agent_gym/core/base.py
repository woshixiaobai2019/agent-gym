import json
import asyncio
import requests
import aiohttp
import signal
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .types import ObservationType, ActionType, ToolsType, Reward, Done
from .exceptions import APIError

class TimeoutError(EnvironmentError):
    """Timeout error for tool execution."""
    pass


@contextmanager
def timeout_context(seconds: int):
    """Context manager for handling timeouts."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class Environment(ABC):
    """Abstract base class for all environments."""
    
    @abstractmethod
    def reset(self) -> Tuple[ObservationType, ToolsType]:
        """Reset the environment and return initial observation and tools."""
        pass
    
    @abstractmethod
    def step(self, action: ActionType, timeout: int = 30) -> Tuple[ObservationType, Reward, Done]:
        """Execute an action in the environment with timeout.
        
        Args:
            action: The action to execute
            timeout: Maximum execution time in seconds (default: 30)
            
        Returns:
            Tuple of (next_observation, reward, done)
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up resources (optional)."""
        pass


class StaticEnvironment(Environment):
    """Base class for environments with predefined tasks."""
    
    def __init__(self, data_file: str = None, task_id: int = 0):
        self.data_file = data_file
        self.task_id = task_id
        self.current_task = None
    
    def _load_task_data(self):
        """Load task data from file."""
        if not self.data_file:
            return
            
        try:
            import json
            with open(self.data_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            if 0 <= self.task_id < len(tasks):
                self.current_task = tasks[self.task_id]
        except Exception as e:
            raise EnvironmentError(f"Failed to load task data: {e}")


class InteractiveEnvironment(Environment):
    """Abstract base class for environments that involve user interaction.
    
    These environments use a UserModel to simulate or handle user responses.
    """
    
    def __init__(self, user_model: 'UserModel'):
        self.user_model = user_model
    
    @abstractmethod
    def _process_user_interaction(self, agent_action: ActionType) -> Tuple[ObservationType, Reward, Done]:
        """Process the interaction between agent and user model.
        
        Args:
            agent_action: The agent's action.
            
        Returns:
            Result of the interaction.
        """
        pass


class UserModel(ABC):
    """Abstract base class for user models."""
    
    @abstractmethod
    def get_initial_query(self) -> ObservationType:
        """Get the initial user query to start the interaction.
        
        Returns:
            The initial user query/request.
        """
        pass
    
    @abstractmethod
    def respond(self, agent_response: ActionType) -> Tuple[Optional[ObservationType], Done]:
        """Respond to agent's action.
        
        Args:
            agent_response: The agent's response/action.
            
        Returns:
            A tuple containing:
            - user_response: User's response to agent (None if no response needed)
            - done: Whether user is satisfied/wants to end interaction
        """
        pass
    
    def reset(self) -> None:
        """Reset the user model's internal state (optional)."""
        pass


class Agent(ABC):
    """Abstract base class for all agents."""
    
    @abstractmethod
    def act(self, observation: ObservationType, tools: Optional[ToolsType] = None) -> ActionType:
        """Generate an action based on the current observation and available tools."""
        pass
    
    def reset(self) -> None:
        """Reset the agent's internal state (optional)."""
        pass


class OpenAICompatibleAgent(Agent):
    """Abstract base class for agents that use OpenAI-compatible API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "dummy", 
                 model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self._conversation_history = []
        self._session = None
    
    def _make_api_call(self, messages: List[Dict[str, Any]], 
                      stream: bool = False, **kwargs) -> Dict[str, Any]:
        """Make synchronous API call to OpenAI-compatible endpoint."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
            **kwargs
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=stream,
                timeout=60
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise APIError(f"API request failed: {e}")
    
    async def _make_api_call_async(self, messages: List[Dict[str, Any]], 
                                  stream: bool = False, **kwargs) -> Dict[str, Any]:
        """Make asynchronous API call to OpenAI-compatible endpoint."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
            **kwargs
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            async with self._session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                
                if stream:
                    return await self._handle_stream_response_async(response)
                else:
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            raise APIError(f"Async API request failed: {e}")
    
    def _make_api_calls_concurrent(self, messages_list: List[List[Dict[str, Any]]], 
                                  max_workers: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Make multiple concurrent API calls."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_messages = {
                executor.submit(self._make_api_call, messages, **kwargs): messages 
                for messages in messages_list
            }
            
            for future in as_completed(future_to_messages):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    raise APIError(f"Concurrent API call failed: {e}")
        
        return results
    
    def _handle_stream_response(self, response) -> Dict[str, Any]:
        """Handle streaming response from API."""
        full_content = ""
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                full_content += delta['content']
                    except json.JSONDecodeError:
                        continue
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": full_content
                }
            }]
        }
    
    async def _handle_stream_response_async(self, response) -> Dict[str, Any]:
        """Handle streaming response from async API."""
        full_content = ""
        
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data.strip() == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            full_content += delta['content']
                except json.JSONDecodeError:
                    continue
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": full_content
                }
            }]
        }
    
    @abstractmethod
    def _convert_observation_to_messages(self, observation: ObservationType) -> List[Dict[str, Any]]:
        """Convert observation to OpenAI messages format."""
        pass
    
    @abstractmethod
    def _convert_tools_to_openai_format(self, tools: ToolsType) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI tools format."""
        pass
    
    @abstractmethod
    def _convert_response_to_action(self, response: Dict[str, Any]) -> ActionType:
        """Convert OpenAI API response to action."""
        pass
    
    def reset(self) -> None:
        """Reset conversation history."""
        self._conversation_history = []
    
    async def close(self):
        """Close async session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._session and not self._session.closed:
            # Try to close session if event loop is running
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.close())
            except RuntimeError:
                pass