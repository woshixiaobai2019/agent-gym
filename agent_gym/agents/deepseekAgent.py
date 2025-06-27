"""DeepSeek Agent implementation using OpenAI-compatible API with Mirau format."""

import json
import re
from typing import Dict, List, Any, Optional
from ..core.base import OpenAICompatibleAgent
from ..core.types import ObservationType, ActionType, ToolsType


class DeepSeekAgent(OpenAICompatibleAgent):
    """DeepSeek Agent that uses deepseek-chat model with Mirau agent format."""
    
    def __init__(self, base_url: str = "https://api.deepseek.com", 
                 api_key: str = "sk-your-deepseek-api-key",
                 system_prompt: str = ""):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_name="deepseek-chat",
            temperature=0.7
        )
        self.custom_system_prompt = system_prompt
        self.tools_info = None
        self._last_tool_calls = []  # Track last tool calls for response formatting
    
    def act(self, observation: ObservationType, tools: Optional[ToolsType] = None) -> ActionType:
        """Generate an action based on observation and tools."""
        if tools is not None:
            self.tools_info = tools
        
        messages = self._convert_observation_to_messages(observation)
        
        # Use streaming API call
        response = self._make_api_call(messages, stream=True)
        
        return self._convert_response_to_action(response)
    
    def _convert_observation_to_messages(self, observation: ObservationType) -> List[Dict[str, Any]]:
        """Convert observation to OpenAI messages format."""
        if not self._conversation_history:
            system_prompt = self._build_system_prompt()
            self._conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
        
        if isinstance(observation, str):
            # Check if this is a tool execution result from Environment
            if self._last_tool_calls and not observation.startswith('<tool_response'):
                # This is Environment's response to tool calls, convert to mirau format
                formatted_responses = self._format_tool_responses(observation)
                self._conversation_history.append({
                    "role": "user",
                    "content": formatted_responses
                })
            else:
                # Regular user message or already formatted tool response
                self._conversation_history.append({
                    "role": "user",
                    "content": observation
                })
        
        return self._conversation_history.copy()
    
    def _format_tool_responses(self, env_response: str) -> str:
        """Format Environment's tool execution response to DeepSeek agent format."""
        if not self._last_tool_calls:
            return env_response
        
        # 简单处理：为每个工具调用创建对应的响应
        if len(self._last_tool_calls) == 1:
            # 单个工具调用，直接包装整个响应
            tool_name = self._last_tool_calls[0]["function"]["name"]
            return f'<tool_response name="{tool_name}">{env_response}</tool_response>'
        else:
            # 多个工具调用，尝试按行分割，如果不够就重复使用最后一行
            lines = env_response.strip().split('\n')
            formatted_responses = []
            
            for i, tool_call in enumerate(self._last_tool_calls):
                tool_name = tool_call["function"]["name"]
                # 如果有对应的行就使用，否则使用整个响应
                response_content = lines[i] if i < len(lines) else env_response
                formatted_responses.append(f'<tool_response name="{tool_name}">{response_content}</tool_response>')
            
            return '\n'.join(formatted_responses)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for DeepSeek agent."""
        # If custom system prompt is provided, use it
        if self.custom_system_prompt:
            return self.custom_system_prompt
        
        # Default system prompt with tools info
        tools_json = json.dumps(
            self.tools_info if self.tools_info else [], 
            ensure_ascii=False, 
            separators=(',', ':')
        )
        
        return f"""你是一个智能AI助手，可以使用提供的工具来完成各种任务。

## 可使用的工具
{tools_json}

## 工具调用格式
当你需要使用工具时，请使用以下XML格式：
<tool_call>
{{"name": "工具名称", "arguments": {{"参数名": "参数值"}}}}
</tool_call>

## 工具响应格式
工具执行结果会以以下格式返回：
<tool_response name="工具名称">工具执行结果</tool_response>

请根据用户的请求选择合适的工具并正确调用。
"""
    
    def _convert_tools_to_openai_format(self, tools: ToolsType) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI tools format."""
        return tools if tools else []
    
    def _convert_response_to_action(self, response: Dict[str, Any]) -> ActionType:
        """Convert OpenAI API response to standard action format."""
        content = response['choices'][0]['message']['content']
        
        self._conversation_history.append({
            "role": "assistant",
            "content": content
        })
        
        tool_calls = self._parse_tool_calls(content)
        
        if tool_calls:
            # Store tool calls for next response formatting
            self._last_tool_calls = tool_calls
            return {"tool_calls": tool_calls}
        else:
            # Clear tool calls tracking
            self._last_tool_calls = []
            return {"content": content}
    
    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse tool calls from DeepSeek agent response (same format as Mirau)."""
        tool_calls = []
        
        # 使用更精确的解析方法
        import re
        
        # 找到所有 <tool_call> 标签的位置
        start_pattern = r'<tool_call>\s*'
        end_pattern = r'\s*</tool_call>'
        
        start_matches = list(re.finditer(start_pattern, content))
        
        for i, start_match in enumerate(start_matches):
            start_pos = start_match.end()
            
            # 找到对应的结束标签
            end_match = re.search(end_pattern, content[start_pos:])
            if not end_match:
                continue
            
            end_pos = start_pos + end_match.start()
            
            # 提取JSON字符串
            json_str = content[start_pos:end_pos].strip()
            
            try:
                tool_data = json.loads(json_str)
                tool_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tool_data["name"],
                        "arguments": json.dumps(
                            tool_data["arguments"], 
                            ensure_ascii=False, 
                            separators=(',', ':')
                        )
                    }
                })
            except json.JSONDecodeError as e:
                print(f"DEBUG: Failed to parse JSON: {json_str}")
                print(f"DEBUG: Error: {e}")
                continue
        
        return tool_calls
    
    def reset(self) -> None:
        """Reset conversation history and tool call tracking."""
        self._conversation_history = []
        self._last_tool_calls = []
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Update custom system prompt."""
        self.custom_system_prompt = system_prompt
        # Reset conversation history to apply new system prompt
        self.reset()