"""Synthesize training data using DeepSeek Agent for Mirau Agent training."""

import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent_gym.envs.CommandLineEnvironment import CommandLineEnvironment
from agent_gym.agents.deepseekAgent import DeepSeekAgent
from agent_gym.runners.base_runner import BaseRunner


class TrainingDataSynthesizer(BaseRunner):
    """Synthesizer for creating Mirau Agent training data using DeepSeek."""
    
    def __init__(self, 
                 data_file: str,
                 deepseek_api_key: str,
                 deepseek_base_url: str = "https://api.deepseek.com",
                 system_prompt: str = "",
                 output_dir: str = "training_data",
                 log_dir: str = "logs/synthesis"):
        # Store synthesizer specific parameters
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_base_url = deepseek_base_url
        self.system_prompt = system_prompt
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Call parent constructor
        super().__init__(data_file, log_dir)
    
    def _create_agent(self):
        return DeepSeekAgent(
            base_url=self.deepseek_base_url,
            api_key=self.deepseek_api_key,
            system_prompt=self.system_prompt
        )
    
    def _create_environment(self, task_id: int):
        return CommandLineEnvironment(data_file=self.data_file, task_id=task_id)
    
    def _get_runner_name(self) -> str:
        return "Training Data Synthesis"
    
    def _get_metadata_extras(self) -> Dict[str, Any]:
        return {
            "deepseek_base_url": self.deepseek_base_url,
            "output_dir": str(self.output_dir)
        }
    
    def synthesize_single_task(self, task_id: int, verbose: bool = True, max_turns: int = 20) -> Dict[str, Any]:
        """Synthesize training data for a single task."""
        if task_id >= len(self.tasks):
            raise ValueError(f"Task ID {task_id} not found (available: 0-{len(self.tasks)-1})")
        
        # Create environment for this task
        env = self._create_environment(task_id)
        
        # Initialize variables
        turn_count = 0
        total_reward = 0.0
        done = False
        start_time = time.time()
        
        task_info = self.tasks[task_id]
        current_user = self._get_current_user()
        
        try:
            if verbose:
                self._print_task_header(task_id, task_info, current_user)
            
            # Reset environment and agent
            observation, tools = env.reset()
            self.agent.reset()
            
            if verbose:
                self._print_initial_observation(observation)
                print(f"🔧 Available tools: {[tool['function']['name'] for tool in tools]}")
            
            # Main interaction loop
            while not done and turn_count < max_turns:
                turn_count += 1
                
                if verbose:
                    print(f"\n--- Turn {turn_count} ---")
                
                try:
                    # Agent takes action
                    action_start_time = time.time()
                    action = self.agent.act(observation, tools if turn_count == 1 else None)
                    action_time = time.time() - action_start_time
                    
                    if verbose:
                        print(f"🔧 PARSED ACTION: {self._format_action_for_display(action)}")
                    
                    # Environment processes action
                    env_start_time = time.time()
                    next_observation, reward, done = env.step(action)
                    env_time = time.time() - env_start_time
                    total_reward += reward
                    
                    if verbose:
                        self._print_environment_response(next_observation, reward, done, total_reward)
                    
                    observation = next_observation
                    
                except Exception as turn_error:
                    error_msg = f"Error in turn {turn_count}: {str(turn_error)}"
                    if verbose:
                        print(f"❌ {error_msg}")
                    break
            
            # Calculate final metrics
            end_time = time.time()
            total_time = end_time - start_time
            success = done and total_reward > 0.5
            
            # Get conversation history from agent (already in correct format!)
            messages = self.agent._conversation_history
            
            # Save training data
            output_file = None
            if len(messages) >= 3:  # At least system, user, assistant
                training_data = {
                    "messages": messages
                }
                
                # Generate filename
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"task_{task_id}_{timestamp}.json"
                output_file = self.output_dir / filename
                
                # Save training data file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(training_data, f, ensure_ascii=False, indent=2)
                
                if verbose:
                    print(f"📊 Training data saved to: {output_file}")
                    print(f"📈 Messages count: {len(messages)}")
                    print(f"📋 Message roles: {[msg['role'] for msg in messages]}")
                    
                    # 显示最后几条消息的内容摘要
                    for i, msg in enumerate(messages[-3:]):
                        content = msg.get('content', '')[:100] + ('...' if len(msg.get('content', '')) > 100 else '')
                        print(f"   {len(messages)-3+i}: {msg['role']} - {content}")
            else:
                if verbose:
                    print(f"⚠️  Not enough messages to save training data ({len(messages)} messages)")
            
            if verbose:
                self._print_task_summary(success, turn_count, total_reward, total_time)
            
            return {
                "success": success,
                "total_turns": turn_count,
                "total_reward": total_reward,
                "total_time": round(total_time, 3),
                "messages_count": len(messages),
                "output_file": str(output_file) if output_file else None
            }
            
        except Exception as main_error:
            if verbose:
                print(f"❌ Task failed: {main_error}")
            
            return {
                "success": False,
                "error": str(main_error),
                "total_turns": turn_count,
                "total_reward": total_reward,
                "total_time": round(time.time() - start_time, 3),
                "messages_count": 0
            }
            
        finally:
            env.cleanup()
    
    def synthesize_all_tasks(self, verbose: bool = True, max_turns: int = 20) -> Dict[str, Any]:
        """Synthesize training data for all tasks."""
        results = []
        overall_start_time = time.time()
        current_user = self._get_current_user()
        
        if verbose:
            print(f"\n🚀 Starting Training Data Synthesis")
            print(f"Total Tasks: {len(self.tasks)}")
            print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Current User: {current_user}")
            print(f"Output Directory: {self.output_dir}")
        
        successful_files = 0
        
        for task_id in range(len(self.tasks)):
            if verbose:
                print(f"\n📋 Synthesizing Task {task_id + 1}/{len(self.tasks)}")
            
            try:
                result = self.synthesize_single_task(task_id, verbose=verbose, max_turns=max_turns)
                results.append(result)
                
                if result.get("success") and result.get("output_file"):
                    successful_files += 1
                    
                # 任务间短暂休息，避免API限制
                if task_id < len(self.tasks) - 1:
                    time.sleep(2)
                    
            except Exception as e:
                if verbose:
                    print(f"❌ Task {task_id} failed: {e}")
                results.append({"success": False, "error": str(e)})
        
        # Calculate overall statistics
        overall_time = time.time() - overall_start_time
        total_messages = sum(r.get("messages_count", 0) for r in results)
        
        summary = {
            "synthesis_type": "DeepSeek -> Mirau Training Data",
            "total_tasks": len(self.tasks),
            "successful_files": successful_files,
            "success_rate": successful_files / len(self.tasks) if self.tasks else 0,
            "total_messages": total_messages,
            "average_messages_per_task": total_messages / len(self.tasks) if self.tasks else 0,
            "total_time": round(overall_time, 3),
            "output_directory": str(self.output_dir),
            "task_results": results
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🏁 SYNTHESIS COMPLETED")
            print(f"Success Rate: {summary['success_rate']:.1%} ({successful_files}/{len(self.tasks)})")
            print(f"Total Training Files: {successful_files}")
            print(f"Total Messages: {total_messages}")
            print(f"Average Messages per Task: {summary['average_messages_per_task']:.1f}")
            print(f"Total Time: {overall_time:.2f}s")
            print(f"Output Directory: {self.output_dir}")
            print(f"{'='*80}")
        
        # Save synthesis summary
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"synthesis_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"📊 Synthesis summary saved to: {summary_file}")
        
        return summary


def main():
    """Main function for training data synthesis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Synthesize Training Data using DeepSeek Agent")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to command line tasks JSON file")
    parser.add_argument("--deepseek-key", type=str, required=True,
                       help="DeepSeek API key")
    parser.add_argument("--deepseek-url", type=str, default="https://api.deepseek.com",
                       help="DeepSeek API base URL")
    parser.add_argument("--system-prompt", type=str, default="",
                       help="Custom system prompt for DeepSeek agent")
    parser.add_argument("--task-id", type=int, default=None,
                       help="Specific task ID to synthesize (if not provided, synthesizes all)")
    parser.add_argument("--max-turns", type=int, default=20,
                       help="Maximum turns per task")
    parser.add_argument("--output-dir", type=str, default="training_data",
                       help="Directory to save training data files")
    parser.add_argument("--log-dir", type=str, default="logs/synthesis",
                       help="Directory to save logs")
    parser.add_argument("--quiet", action="store_true",
                       help="Run in quiet mode (less verbose output)")
    
    args = parser.parse_args()
    
    # Validate data file exists
    if not Path(args.data_file).exists():
        print(f"❌ Data file not found: {args.data_file}")
        return
    
    # Default system prompt if not provided - 重要：要匹配环境的工具名称！
    if not args.system_prompt:
        args.system_prompt = """You are an intelligent command-line assistant that excels at using Linux commands to complete tasks.

    ## Output Format Requirements
    You must follow this format:
    1. First, use <think> tags for analysis
    2. Then make tool calls or provide final response
    3. You can call multiple tools at once

    <think type="thinking_type">
    Your thought process
    </think>

    Thinking Types:
    - complex: Deep analysis, 500-2000 tokens, requires thorough thinking and multi-step planning
    - mid: Medium analysis, 100-500 tokens, needs analysis but relatively simple
    - quick: Quick judgment, 0-20 tokens, simple and direct operations

    ## Tool Call Format
    <tool_call>
    {"name": "tool_name", "arguments": {"parameter": "value"}}
    </tool_call>

    ## Available Tools
    You can use the following tools:
    - execute_shell: Execute shell commands
    - read_file: Read file contents
    - write_file: Write content to files
    - list_directory: List directory contents
    - create_directory: Create directories
    - delete_file: Delete files
    - move_file: Move/rename files
    - copy_file: Copy files

    ## Examples

    ### Example 1: Complex Task
    User: Find all Python files in the current directory, count total lines of code (excluding comments and empty lines), and create a summary report

    <think type="complex">
    This is a complex task that requires multiple steps:
    1. First, I need to find all Python files in the current directory and subdirectories
    2. For each Python file, count lines of code excluding comments (lines starting with #) and empty lines
    3. Sum up all the line counts
    4. Create a comprehensive summary report with file details

    My approach:
    - Use find command to locate all .py files
    - Use grep and wc to count non-comment, non-empty lines
    - Process each file and collect statistics
    - Generate a formatted report with totals

    Let me start by finding all Python files and then process them systematically.
    </think>

    <tool_call>
    {"name": "execute_shell", "arguments": {"command": "find . -name '*.py' -type f"}}
    </tool_call>

    <tool_call>
    {"name": "execute_shell", "arguments": {"command": "find . -name '*.py' -exec wc -l {} +"}}
    </tool_call>

    ### Example 2: Medium Task
    User: Check disk usage of the current directory and identify the largest files

    <think type="mid">
    This task requires checking disk usage and finding large files:
    1. Get overall disk usage of current directory
    2. Find the largest files to identify space consumers
    3. Present the information in a readable format

    I'll use du command for directory usage and find with sort for largest files.
    </think>

    <tool_call>
    {"name": "execute_shell", "arguments": {"command": "du -sh ."}}
    </tool_call>

    <tool_call>
    {"name": "execute_shell", "arguments": {"command": "find . -type f -exec ls -lh {} + | sort -k5 -hr | head -10"}}
    </tool_call>

    ### Example 3: Simple Task
    User: Create a backup directory and copy all .txt files to it

    <think type="quick">
    Create directory, then copy files
    </think>

    <tool_call>
    {"name": "create_directory", "arguments": {"dir_path": "backup"}}
    </tool_call>

    <tool_call>
    {"name": "execute_shell", "arguments": {"command": "cp *.txt backup/ 2>/dev/null || echo 'No .txt files found'"}}
    </tool_call>

    ## Tool Response Format
    Tool execution results will be returned in this format:
    <tool_response name="tool_name">tool execution result</tool_response>

    ## Guidelines
    1. Choose appropriate thinking type based on task complexity
    2. Make your thinking specific and logical, explaining your analysis and execution plan
    3. Use multiple related tools efficiently when beneficial
    4. Ensure command safety with proper error handling (e.g., 2>/dev/null)
    5. Provide clear summaries and result explanations after completing tasks
    6. Consider edge cases and provide robust solutions
    """
    
    # Initialize synthesizer
    synthesizer = TrainingDataSynthesizer(
        data_file=args.data_file,
        deepseek_api_key=args.deepseek_key,
        deepseek_base_url=args.deepseek_url,
        system_prompt=args.system_prompt,
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )
    
    verbose = not args.quiet
    
    try:
        if args.task_id is not None:
            # Synthesize specific task
            print(f"🎯 Synthesizing specific task: {args.task_id}")
            result = synthesizer.synthesize_single_task(args.task_id, verbose=verbose, max_turns=args.max_turns)
            if result.get("success"):
                print(f"\n✅ Task synthesized successfully. Output: {result.get('output_file')}")
            else:
                print(f"\n❌ Task synthesis failed: {result.get('error', 'Unknown error')}")
        else:
            # Synthesize all tasks
            print(f"🚀 Synthesizing all tasks from: {args.data_file}")
            summary = synthesizer.synthesize_all_tasks(verbose=verbose, max_turns=args.max_turns)
            print(f"\n🏁 Synthesis completed. Success rate: {summary['success_rate']:.1%}")
            print(f"📁 Training files created: {summary['successful_files']}")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Synthesis interrupted by user")
    except Exception as e:
        print(f"\n❌ Synthesis failed: {e}")


if __name__ == "__main__":
    main()