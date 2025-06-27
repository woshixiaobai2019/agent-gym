# Agent Gym - Agent Training Evaluation & Data Synthesis Framework

A highly abstract framework for **training and evaluating intelligent agents** and **synthesizing high-quality training data**. Easily implement custom environments through unified interfaces to generate multi-turn tool calling data.

> 🎯 **Open Source Purpose**: Many people want to know how [mirau-agent](https://huggingface.co/eliuakk/mirau-agent-base-oai) was trained and how to synthesize such high-quality data. This framework is the answer.

## 🏗️ Core Architecture

```
┌─────────────────┐  Standard API  ┌─────────────────┐
│     Agent       │ ←──────────→ │  Environment    │
│ (Any Implementation) │              │ (Any Implementation) │
└─────────────────┘              └─────────────────┘
         ↓                                ↓
┌─────────────────┐              ┌─────────────────┐
│  BaseRunner     │ ←──────────→ │ Complete Trajectory │
│ (Unified Exec)  │              │ Structured Logs  │
└─────────────────┘              └─────────────────┘
         ↓
┌─────────────────┐
│ Training Data   │ → OpenAI Messages Format
│ Synthesizer     │   For Model Training
└─────────────────┘
```

**Two Core Functions**:
1. **Training & Evaluation**: Test agent capabilities in standardized environments, record complete interaction trajectories
2. **Data Synthesis**: Use powerful models to generate high-quality data for training weaker models

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/woshixiaobai2019/agent-gym.git
cd agent-gym
pip install requests aiohttp
```

### Training & Evaluation Mode

Test your agent's performance across various environments:

```bash
# Command-line environment testing
python eval/main_cmd.py --data-file agent_gym/data/cmd.json --task-id 0

# Python environment testing
python eval/main_python.py --data-file agent_gym/data/py.json --task-id 0

# NLP dialogue environment testing
python eval/main_nlp.py --data-file agent_gym/data/nlp.json --task-id 0
```

Automatically generates detailed JSON logs including:
- Complete dialogue trajectories
- Tool calling sequences
- Performance metrics and success rates
- Time consumption statistics

### Data Synthesis Mode

**This is how mirau-agent's training data was generated**:

```bash
# Synthesize training data using DeepSeek
python synthesizer/trainingDataSynthesizer.py \
  --data-file agent_gym/data/cmd.json \
  --deepseek-key "your-deepseek-api-key" \
  --output-dir "training_data"

# Batch synthesize data for all environments
for file in agent_gym/data/*.json; do
    python synthesizer/trainingDataSynthesizer.py \
        --data-file "$file" \
        --deepseek-key "your-key" \
        --output-dir "training_data/$(basename "$file" .json)"
done
```

## 📊 Generated Training Data Format

Standard OpenAI Messages format, ready for model training:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "[{\"type\":\"function\",\"function\":{\"name\":\"execute_shell\",...}}]"
    },
    {
      "role": "user", 
      "content": "Find all Python files in the current directory"
    },
    {
      "role": "assistant",
      "content": "<think type=\"quick\">\nSimple file search operation\n</think>\n\n<tool_call>\n{\"name\": \"execute_shell\", \"arguments\": {\"command\": \"find . -name '*.py' -type f\"}}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "<tool_response name=\"execute_shell\">./test.py\n./main.py</tool_response>"
    }
  ]
}
```

## 🤖 Built-in Agent Types

### Mirau Agent (Target Training Model)
- Uses `<tool_call>` XML format
- Supports multi-turn dialogue and complex reasoning
- Local/remote API deployment

### DeepSeek Agent (Data Generation Model)  
- Supports `<think>` reasoning process
- High-quality data generation
- Standardized output format

## 🌍 Built-in Environment Types

| Environment | Description | Use Cases |
|------------|-------------|-----------|
| **CommandLine** | Linux command execution, file operations | System admin, DevOps, file processing |
| **Python** | Safe Python code execution | Math computation, data analysis, algorithms |
| **NLP** | LLM-based dialogue interaction | Customer service, content creation, role-play |

## 💡 Custom Environments

Create your environment in three steps:

```python
# 1. Inherit base environment
from agent_gym.core.base import StaticEnvironment

class MyEnvironment(StaticEnvironment):
    def reset(self):
        return "Task description", self.tools
    
    def step(self, action):
        # Handle agent actions
        return "Environment response", reward, done

# 2. Create runner
from agent_gym.runners.base_runner import BaseRunner

class MyRunner(BaseRunner):
    def _create_environment(self, task_id):
        return MyEnvironment(self.data_file, task_id)

# 3. Prepare task data JSON and run
```

## 🎯 Core Value

### For Model Training
- **Standardized Evaluation**: Unified environment interfaces for fair agent comparison
- **Detailed Trajectories**: Complete interaction records for analysis and improvement
- **Batch Processing**: Large-scale task evaluation support

### For Data Synthesis
- **High-Quality Data**: Generate training data for weaker models using powerful models
- **Reasoning Process**: Complete reasoning chains to improve training effectiveness
- **Standard Format**: Direct compatibility with mainstream training frameworks

## 📁 Project Structure

```
agent_gym/
├── core/          # Core abstraction layer (Agent, Environment, Types)
├── agents/        # Agent implementations (Mirau, DeepSeek)
├── envs/          # Environment implementations (CommandLine, Python, NLP)
├── runners/       # Executors (unified execution logic)
├── data/          # Sample task data
└── synthesizer/   # Training data synthesizer
```

## 🔗 Related Resources

- **mirau-agent model**: [HuggingFace](https://huggingface.co/eliuakk/mirau-agent-base-oai)
- **DeepSeek API**: [Official Platform](https://platform.deepseek.com/)
- **OpenAI Function Calling**: [API Documentation](https://platform.openai.com/docs/guides/function-calling)
- **python sandbox**: [SandboxFusion](https://github.com/bytedance/SandboxFusion)
---

**MIT License** - Use freely, maintain yourself 🚀

> **Note**: This framework is open-sourced primarily to share mirau-agent's training methodology and data synthesis techniques. Please maintain and use at your own discretion.