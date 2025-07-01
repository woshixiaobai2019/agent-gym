# Enhanced Command Line Environment

This document provides an overview of the `EnhancedCommandLineEnvironment` implementation.

## Overview

The `EnhancedCommandLineEnvironment` extends the base `CommandLineEnvironment` to support a new data format with additional features while maintaining full backward compatibility.

## Key Features

### 1. Enhanced Data Format Support
- **create_env**: Python script to create the main test environment
- **create_noise_env_appendix**: Additional script to create noise/distraction files
- **Enhanced task structure** with:
  - `level`: Difficulty level (simple, medium, hard)
  - `question`: The main task question
  - `format_example`: Example of expected output format
  - `allowed_commands`: List of allowed shell commands (empty means all allowed)
  - `reference_answer`: Reference command/solution
  - `verify_answer`: Python code to verify the answer

### 2. Command Filtering
- Restricts shell command execution based on `allowed_commands` list
- Empty list means all commands are allowed
- Provides clear error messages for disallowed commands

### 3. Noise Environment
- Creates additional files and directories to test agent robustness
- Helps simulate real-world environments with distracting elements

### 4. Enhanced Verification
- Uses `verify_answer` field for more sophisticated task completion verification
- Maintains backward compatibility with legacy `verify` field

### 5. Backward Compatibility
- Fully compatible with existing `CommandLineEnvironment` data format
- Can be used as a drop-in replacement
- Automatically detects and handles both old and new data formats

### 6. Enhanced Logging and Status
- Comprehensive logging with different levels
- Enhanced workspace status printing with task details
- Better error handling and debugging information

## Data Format Examples

### Enhanced Format
```json
{
  "create_env": "import os\nos.makedirs('factory_data', exist_ok=True)...",
  "create_noise_env_appendix": "# 干扰文件\nwith open('personal/shopping_list.txt', 'w') as f:...",
  "task": {
    "level": "simple", 
    "question": "查看工厂8月份生产的圣诞装饰品总件数是多少？将结果写入result.txt（只输出一个整数）",
    "format_example": "5400",
    "allowed_commands": ["grep", "awk", "cat", "echo"],
    "reference_answer": "grep '圣诞装饰品' factory_data/production_august.log | awk '{sum+=$3} END {print sum}' > result.txt",
    "verify_answer": "try:\n    with open('result.txt', 'r') as f:\n        result = f.read().strip()\n    expected = '5400'\n    success = (result == expected)\nexcept:\n    success = False"
  }
}
```

### Legacy Format (Still Supported)
```json
{
  "query": "统计workspace目录下所有.txt文件里字母'R'（大小写）的总出现次数，将结果写入result.txt文件中(最终结果只写数字)",
  "env": "# 创建workspace目录和测试文件\nimport os\nos.makedirs('workspace', exist_ok=True)...",
  "verify": "# 验证结果\ntry:\n    with open('result.txt', 'r') as f:\n        agent_result = f.read().strip()..."
}
```

## Usage

```python
from agent_gym.envs.EnhancedCommandLineEnvironment import EnhancedCommandLineEnvironment

# Create environment with enhanced data
env = EnhancedCommandLineEnvironment(data_file='enhanced_tasks.json', task_id=0)

# Reset environment
observation, tools = env.reset()

# Execute actions
action = {
    "tool_calls": [{
        "function": {
            "name": "execute_shell",
            "arguments": '{"command": "ls"}'
        }
    }]
}

result, reward, done = env.step(action)

# Get task information
task_info = env.get_task_info()
print(task_info)

# Clean up
env.cleanup()
```

## Additional Methods

### `get_task_info()`
Returns detailed information about the current task:
- Format type (enhanced/legacy)
- Task level, question, format example
- Allowed commands, reference answer
- Workspace directory location
- Environment setup details

## Testing

The implementation has been thoroughly tested with:
- ✅ Basic functionality tests
- ✅ Enhanced format tests
- ✅ Legacy format compatibility tests
- ✅ Command filtering tests
- ✅ Drop-in replacement tests
- ✅ Comprehensive feature tests

All tests pass successfully, confirming that the `EnhancedCommandLineEnvironment` provides all the required functionality while maintaining full backward compatibility.