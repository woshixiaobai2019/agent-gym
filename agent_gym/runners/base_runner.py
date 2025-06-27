"""Base Runner for Agent Gym environments."""

import json
import time
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..core.base import Environment,Agent
from ..core.exceptions import EnvironmentError


class BaseRunner(ABC):
    """Abstract base class for all environment runners."""
    
    def __init__(self, 
                 data_file: str,
                 log_dir: str = "logs",
                 **kwargs):
        """Initialize the base runner.
        
        Args:
            data_file: Path to JSON file containing tasks
            log_dir: Directory to save logs
            **kwargs: Additional parameters for specific runners
        """
        self.data_file = data_file
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.kwargs = kwargs
        
        # Initialize agent (delegated to subclass)
        self.agent = self._create_agent()
        
        # Load tasks
        self.tasks = self._load_tasks()
    
    @abstractmethod
    def _create_agent(self) -> Agent:
        """Create and return agent instance."""
        pass
    
    @abstractmethod
    def _create_environment(self, task_id: int) -> Environment:
        """Create and return environment instance for specific task."""
        pass
    
    @abstractmethod
    def _get_runner_name(self) -> str:
        """Return the name of this runner for logging purposes."""
        pass
    
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from data file."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise EnvironmentError(f"Failed to load tasks: {e}")
    
    def _get_current_user(self) -> str:
        """Get current user name from environment or system."""
        # Try environment variable first
        user = os.environ.get('USER') or os.environ.get('USERNAME')
        if user:
            return user
        
        # Fallback to system methods
        try:
            import getpass
            return getpass.getuser()
        except:
            return "unknown_user"
    
    def run_single_task(self, task_id: int, verbose: bool = True, max_turns: int = 20) -> Dict[str, Any]:
        """Run a single task with detailed logging."""
        if task_id >= len(self.tasks):
            raise ValueError(f"Task ID {task_id} not found (available: 0-{len(self.tasks)-1})")
        
        # Create environment for this task
        env = self._create_environment(task_id)
        
        # Initialize variables
        turn_count = 0
        total_reward = 0.0
        done = False
        start_time = time.time()
        
        # Prepare logging
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        runner_name = self._get_runner_name().lower().replace(' ', '_')
        log_file = self.log_dir / f"{runner_name}_task_{task_id}_{timestamp}.json"
        
        task_info = self.tasks[task_id]
        current_user = self._get_current_user()
        
        # Initialize log data
        log_data = {
            "metadata": {
                "runner_type": self._get_runner_name(),
                "task_id": task_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "user": current_user,
                "max_turns": max_turns,
                **self._get_metadata_extras()
            },
            "task_config": task_info,
            "trajectory": [],
            "summary": {}
        }
        
        try:
            if verbose:
                self._print_task_header(task_id, task_info, current_user)
            
            # Reset environment
            observation, tools = env.reset()
            
            if verbose:
                self._print_initial_observation(observation)
            
            # Log initial observation
            log_data["trajectory"].append({
                "turn": 0,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "type": "environment_response",
                "content": observation,
                "metadata": {
                    "response_type": "initial",
                    "tools_available": len(tools) if tools else 0
                }
            })
            
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
                    
                    # Format agent action for display
                    action_display = self._format_action_for_display(action)
                    
                    if verbose:
                        print(f"🤖 AGENT: {action_display}")
                    
                    # Log agent action
                    log_data["trajectory"].append({
                        "turn": turn_count,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "type": "agent_action",
                        "content": action,
                        "display": action_display,
                        "metadata": {
                            "processing_time": round(action_time, 3),
                            "action_type": self._get_action_type(action)
                        }
                    })
                    
                    # Environment processes action
                    env_start_time = time.time()
                    next_observation, reward, done = env.step(action)
                    env_time = time.time() - env_start_time
                    total_reward += reward
                    
                    if verbose:
                        self._print_environment_response(next_observation, reward, done, total_reward)
                    
                    # Log environment response
                    log_data["trajectory"].append({
                        "turn": turn_count,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "type": "environment_response",
                        "content": next_observation,
                        "metadata": {
                            "processing_time": round(env_time, 3),
                            "reward": reward,
                            "done": done,
                            "total_reward": total_reward
                        }
                    })
                    
                    observation = next_observation
                    
                except Exception as turn_error:
                    error_msg = f"Error in turn {turn_count}: {str(turn_error)}"
                    if verbose:
                        print(f"❌ {error_msg}")
                    
                    # Log error
                    log_data["trajectory"].append({
                        "turn": turn_count,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "type": "error",
                        "content": error_msg,
                        "metadata": {
                            "error_type": type(turn_error).__name__
                        }
                    })
                    break
            
            # Calculate final metrics
            end_time = time.time()
            total_time = end_time - start_time
            
            # Determine success
            success = done and total_reward > 0.5
            
            # Log summary
            log_data["summary"] = {
                "success": success,
                "total_turns": turn_count,
                "total_reward": total_reward,
                "total_time": round(total_time, 3),
                "avg_time_per_turn": round(total_time / max(turn_count, 1), 3),
                "completion_reason": self._get_completion_reason(success, turn_count, max_turns)
            }
            
            if verbose:
                self._print_task_summary(success, turn_count, total_reward, total_time)
            
            # Save log
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            if verbose:
                print(f"📝 Log saved to: {log_file}")
            
            return log_data["summary"]
            
        except Exception as main_error:
            error_summary = {
                "success": False,
                "error": str(main_error),
                "total_turns": turn_count,
                "total_reward": total_reward,
                "total_time": round(time.time() - start_time, 3)
            }
            
            log_data["summary"] = error_summary
            
            # Save error log
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            if verbose:
                print(f"❌ Task failed: {main_error}")
                print(f"📝 Error log saved to: {log_file}")
            
            return error_summary
            
        finally:
            env.cleanup()
    
    def run_all_tasks(self, verbose: bool = True, max_turns: int = 20) -> Dict[str, Any]:
        """Run all tasks and return aggregated results."""
        results = []
        overall_start_time = time.time()
        current_user = self._get_current_user()
        
        if verbose:
            print(f"\n🚀 Starting {self._get_runner_name()} Test Suite")
            print(f"Total Tasks: {len(self.tasks)}")
            print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Current User: {current_user}")
        
        for task_id in range(len(self.tasks)):
            if verbose:
                print(f"\n📋 Running Task {task_id + 1}/{len(self.tasks)}")
            
            try:
                result = self.run_single_task(task_id, verbose=verbose, max_turns=max_turns)
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"❌ Task {task_id} failed: {e}")
                results.append({"success": False, "error": str(e)})
        
        # Calculate overall statistics
        overall_time = time.time() - overall_start_time
        successful_tasks = sum(1 for r in results if r.get("success", False))
        total_reward = sum(r.get("total_reward", 0) for r in results)
        
        summary = {
            "runner_type": self._get_runner_name(),
            "total_tasks": len(self.tasks),
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / len(self.tasks) if self.tasks else 0,
            "total_reward": total_reward,
            "average_reward": total_reward / len(self.tasks) if self.tasks else 0,
            "total_time": round(overall_time, 3),
            "task_results": results
        }
        
        if verbose:
            self._print_final_summary(summary)
        
        # Save overall summary
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        runner_name = self._get_runner_name().lower().replace(' ', '_')
        summary_file = self.log_dir / f"{runner_name}_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"📊 Summary saved to: {summary_file}")
        
        return summary
    
    # Display methods that can be overridden by subclasses
    def _print_task_header(self, task_id: int, task_info: Dict[str, Any], current_user: str):
        """Print task header information."""
        print(f"\n{'='*80}")
        print(f"{self._get_runner_name().upper()} TASK EXECUTION")
        print(f"Task ID: {task_id}")
        print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User: {current_user}")
        print(f"{'='*80}")
        self._print_task_details(task_info)
        print(f"{'='*80}")
    
    def _print_task_details(self, task_info: Dict[str, Any]):
        """Print task-specific details. Override in subclasses."""
        for key, value in task_info.items():
            if isinstance(value, str) and len(value) < 200:
                print(f"{key.title()}: {value}")
    
    def _print_initial_observation(self, observation: str):
        """Print initial observation."""
        print(f"\n📝 INITIAL: {observation}")
    
    def _print_environment_response(self, observation: str, reward: float, done: bool, total_reward: float):
        """Print environment response."""
        print(f"🌍 ENV: {observation}")
        print(f"📊 Reward: {reward}, Done: {done}, Total Reward: {total_reward}")
    
    def _print_task_summary(self, success: bool, turn_count: int, total_reward: float, total_time: float):
        """Print task completion summary."""
        print(f"\n{'='*80}")
        print(f"TASK COMPLETED")
        print(f"Success: {'✅' if success else '❌'}")
        print(f"Total Turns: {turn_count}")
        print(f"Total Reward: {total_reward}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"{'='*80}")
    
    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final summary for all tasks."""
        print(f"\n{'='*80}")
        print(f"🏁 ALL TASKS COMPLETED")
        print(f"Success Rate: {summary['success_rate']:.1%} ({summary['successful_tasks']}/{summary['total_tasks']})")
        print(f"Average Reward: {summary['average_reward']:.3f}")
        print(f"Total Time: {summary['total_time']:.2f}s")
        print(f"{'='*80}")
    
    # Utility methods that can be overridden
    def _format_action_for_display(self, action: Dict[str, Any]) -> str:
        """Format agent action for readable display."""
        if action.get("tool_calls"):
            tool_strs = []
            for tool_call in action["tool_calls"]:
                func_name = tool_call["function"]["name"]
                try:
                    func_args = json.loads(tool_call["function"]["arguments"])
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in func_args.items())
                    tool_strs.append(f"{func_name}({args_str})")
                except:
                    tool_strs.append(f"{func_name}(invalid_args)")
            return f"🔧 {'; '.join(tool_strs)}"
        elif action.get("content"):
            return f"💬 {action['content']}"
        else:
            return f"📝 {str(action)}"
    
    def _get_action_type(self, action: Dict[str, Any]) -> str:
        """Get action type for logging."""
        if action.get("tool_calls"):
            return "tool_calls"
        elif action.get("content"):
            return "message"
        else:
            return "unknown"
    
    def _get_completion_reason(self, success: bool, turn_count: int, max_turns: int) -> str:
        """Get completion reason for logging."""
        if success:
            return "success"
        elif turn_count >= max_turns:
            return "max_turns"
        else:
            return "error"
    
    def _get_metadata_extras(self) -> Dict[str, Any]:
        """Get additional metadata for logging. Override in subclasses."""
        return {}