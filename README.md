# Agent Gym - Agent训练评估与数据合成框架

一个高度抽象的框架，用于**训练评估智能Agent**和**合成高质量训练数据**。通过统一接口，轻松实现自定义环境，生成多轮工具调用数据。

> 🎯 **开源目的**: 很多人想知道 [mirau-agent](https://huggingface.co/eliuakk/mirau-agent-base-oai) 是怎么训练的，如何合成这样的高质量数据。这个框架就是答案。

## 🏗️ 核心架构

```
┌─────────────────┐    标准接口    ┌─────────────────┐
│     Agent       │ ←──────────→ │  Environment    │
│   (任意实现)     │              │   (任意实现)     │
└─────────────────┘              └─────────────────┘
         ↓                                ↓
┌─────────────────┐              ┌─────────────────┐
│  BaseRunner     │ ←──────────→ │  完整轨迹记录    │
│   (统一执行)     │              │   结构化日志     │
└─────────────────┘              └─────────────────┘
         ↓
┌─────────────────┐
│ 训练数据合成器   │ → OpenAI Messages Format
│ (DeepSeek等)    │   用于模型训练
└─────────────────┘
```

**两大核心功能**:
1. **训练评估**: 在标准化环境中测试Agent能力，记录完整交互轨迹
2. **数据合成**: 使用强大模型生成训练弱模型的高质量数据

## 🚀 快速开始

### 安装
```bash
git clone https://github.com/woshixiaobai2019/agent-gym.git
cd agent-gym
pip install requests aiohttp
```

### 训练评估模式

测试你的Agent在各种环境中的表现：

```bash
# 命令行环境测试
python eval/main_cmd.py --data-file agent_gym/data/cmd.json --task-id 0

# Python环境测试
python eval/main_python.py --data-file agent_gym/data/py.json --task-id 0

# NLP对话环境测试
python eval/main_nlp.py --data-file agent_gym/data/nlp.json --task-id 0
```

自动生成详细的JSON日志，包含：
- 完整的对话轨迹
- 工具调用序列
- 性能指标和成功率
- 时间消耗统计

### 数据合成模式

**这就是mirau-agent的训练数据来源**：

```bash
# 使用DeepSeek合成训练数据
python synthesizer/trainingDataSynthesizer.py \
  --data-file agent_gym/data/cmd.json \
  --deepseek-key "your-deepseek-api-key" \
  --output-dir "training_data"

# 批量合成所有环境数据
for file in agent_gym/data/*.json; do
    python synthesizer/trainingDataSynthesizer.py \
        --data-file "$file" \
        --deepseek-key "your-key" \
        --output-dir "training_data/$(basename "$file" .json)"
done
```

## 📊 生成的训练数据格式

标准OpenAI Messages格式，直接可用于模型训练：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "[{\"type\":\"function\",\"function\":{\"name\":\"execute_shell\",...}}]"
    },
    {
      "role": "user", 
      "content": "查找当前目录下所有Python文件"
    },
    {
      "role": "assistant",
      "content": "<think type=\"quick\">\n简单的文件查找操作\n</think>\n\n<tool_call>\n{\"name\": \"execute_shell\", \"arguments\": {\"command\": \"find . -name '*.py' -type f\"}}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "<tool_response name=\"execute_shell\">./test.py\n./main.py</tool_response>"
    }
  ]
}
```

## 🤖 内置Agent类型

### Mirau Agent (目标训练模型)
- 使用`<tool_call>`XML格式
- 支持多轮对话和复杂推理
- 本地/远程API部署

### DeepSeek Agent (数据生成模型)  
- 支持`<think>`思考过程
- 高质量数据生成
- 规范化输出格式

## 🌍 内置环境类型

| 环境类型 | 功能描述 | 适用场景 |
|---------|---------|---------|
| **CommandLine** | Linux命令执行、文件操作 | 系统管理、DevOps、文件处理 |
| **Python** | 安全Python代码执行 | 数学计算、数据分析、算法验证 |
| **NLP** | 基于LLM的对话交互 | 客服场景、内容创作、角色扮演 |

## 💡 自定义环境

三步创建你的环境：

```python
# 1. 继承基础环境
from agent_gym.core.base import StaticEnvironment

class MyEnvironment(StaticEnvironment):
    def reset(self):
        return "任务描述", self.tools
    
    def step(self, action):
        # 处理Agent动作
        return "环境响应", reward, done

# 2. 创建Runner
from agent_gym.runners.base_runner import BaseRunner

class MyRunner(BaseRunner):
    def _create_environment(self, task_id):
        return MyEnvironment(self.data_file, task_id)

# 3. 准备任务数据JSON，运行即可
```

## 🎯 核心价值

### 对于模型训练
- **标准化评估**: 统一的环境接口，公平对比不同Agent
- **详细轨迹**: 完整记录交互过程，便于分析改进
- **批量处理**: 支持大规模任务批量评估

### 对于数据合成
- **高质量数据**: 通过强大模型生成训练弱模型的数据
- **思考过程**: 包含完整的推理链，提升训练效果  
- **格式标准**: 直接兼容主流训练框架

## 📁 项目结构

```
agent_gym/
├── core/          # 核心抽象层 (Agent, Environment, Types)
├── agents/        # Agent实现 (Mirau, DeepSeek)
├── envs/          # 环境实现 (CommandLine, Python, NLP)
├── runners/       # 执行器 (统一运行逻辑)
├── data/          # 示例任务数据
└── synthesizer/   # 训练数据合成器
```

## 🔗 相关资源

- **mirau-agent模型**: [HuggingFace](https://huggingface.co/eliuakk/mirau-agent-base-oai)
- **DeepSeek API**: [官方平台](https://platform.deepseek.com/)
- **OpenAI Function Calling**: [API文档](https://platform.openai.com/docs/guides/function-calling)
- **python sandbox**: [SandboxFusion](https://github.com/bytedance/SandboxFusion)
---

**MIT License** - 自由使用，自行维护 🚀

> **说明**: 这个框架开源的主要目的是分享mirau-agent的训练方法和数据合成技术。后续不会提供技术支持，请自行维护使用。