"""Command Line Environment for Agent Gym."""

import os
import subprocess
import tempfile
import shutil
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path

from ..core.base import StaticEnvironment, timeout_context, TimeoutError
from ..core.types import ObservationType, ActionType, ToolsType, Reward, Done
from ..core.exceptions import EnvironmentError


class CommandLineEnvironment(StaticEnvironment):
    """Command line environment that provides file system and shell tools."""
    
    def __init__(self, data_file: str = None, task_id: int = 0):
        """Initialize the command line environment.
        
        Args:
            data_file: Path to JSON file containing tasks
            task_id: Index of the task to load from data file
        """
        super().__init__(data_file, task_id)
        self.workspace_dir = None
        self.tools = self._define_tools()
        
        if data_file:
            self._load_task_data()
    
    def reset(self) -> Tuple[ObservationType, ToolsType]:
        """Reset the environment and set up workspace."""
        # Clean up previous workspace if exists
        self.cleanup()
        
        # Create new temporary workspace directory
        self.workspace_dir = tempfile.mkdtemp(prefix="agent_gym_")
        
        # Setup environment if specified in task
        if self.current_task and "env" in self.current_task:
            self._setup_environment(self.current_task["env"])
        self._print_workspace_status()
        # Return initial query and tools
        query = self.current_task["query"] if self.current_task else "Hello! How can I help you?"
        return query, self.tools
    
    def cleanup(self):
        """Clean up the workspace directory."""
        if self.workspace_dir and os.path.exists(self.workspace_dir):
            try:
                shutil.rmtree(self.workspace_dir)
                self.workspace_dir = None
            except Exception as e:
                print(f"Warning: Failed to cleanup workspace: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()
    
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
                    reward = 1.0 if self._verify_task_completion() else 0.0
                    done = True
                
                return observation, reward, done
                
        except TimeoutError as e:
            return f"Timeout error: {str(e)}", -0.5, False
        except Exception as e:
            return f"Execution error: {str(e)}", -0.1, False
    
    def _setup_environment(self, env_script: str):
        """Set up the environment using the provided script."""
        original_cwd = os.getcwd()
        try:
            os.chdir(self.workspace_dir)
            # Execute the environment setup script
            exec(env_script, {"os": os, "__workspace__": self.workspace_dir})
        finally:
            os.chdir(original_cwd)
    
    def _is_tool_call(self, action: ActionType) -> bool:
        """Check if the action is a tool call."""
        if isinstance(action, dict):
            return "tool_calls" in action and action["tool_calls"]
        return False
    
    def _execute_tool_call(self, action: ActionType) -> str:
        """Execute a tool call and return the result."""
        original_cwd = os.getcwd()
        try:
            os.chdir(self.workspace_dir)
            
            # Parse tool call (assuming OpenAI format)
            tool_calls = action.get("tool_calls", [])
            results = []
            
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                result = self._call_function(function_name, arguments)
                results.append(f"{result}")
            
            return "\n".join(results)
            
        finally:
            os.chdir(original_cwd)
    
    def _evaluate_final_answer(self, action: ActionType) -> str:
        """Evaluate agent's final answer."""
        content = action.get("content", "No answer provided")
        return f"Agent's final answer: {content}"
    
    def _verify_task_completion(self) -> bool:
        """Verify if the task has been completed successfully."""
        if not self.current_task or "verify" not in self.current_task:
            return False
        
        try:
            original_cwd = os.getcwd()
            os.chdir(self.workspace_dir)
            
            # Execute verification script
            local_vars = {"os": os, "__workspace__": self.workspace_dir}
            exec(self.current_task["verify"], local_vars)
            
            # Verification script should set 'success' variable
            return local_vars.get("success", False)
            
        except Exception as e:
            print(f"Verification error: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def _call_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Call a specific function with arguments."""
        if function_name == "read_file":
            return self._read_file(arguments["file_path"])
        elif function_name == "write_file":
            return self._write_file(arguments["file_path"], arguments["content"])
        elif function_name == "list_directory":
            return self._list_directory(arguments.get("dir_path", "."))
        elif function_name == "execute_shell":
            return self._execute_shell(arguments["command"])
        elif function_name == "create_directory":
            return self._create_directory(arguments["dir_path"])
        elif function_name == "delete_file":
            return self._delete_file(arguments["file_path"])
        elif function_name == "move_file":
            return self._move_file(arguments["source_path"], arguments["dest_path"])
        elif function_name == "copy_file":
            return self._copy_file(arguments["source_path"], arguments["dest_path"])
        else:
            raise EnvironmentError(f"Unknown function: {function_name}")
    
    def _read_file(self, file_path: str) -> str:
        """Read file contents."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _write_file(self, file_path: str, content: str) -> str:
        """Write content to file."""
        try:
            # Handle case where file_path is empty or None
            if not file_path:
                return "Error writing file: file path is empty"
            
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File {file_path} written successfully"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def _list_directory(self, dir_path: str) -> str:
        """List directory contents."""
        try:
            items = os.listdir(dir_path)
            return "\n".join(items) if items else "Directory is empty"
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def _execute_shell(self, command: str) -> str:
        """Execute shell command with built-in timeout."""
        try:
            # subprocess has its own timeout mechanism
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=25,  # Slightly less than step timeout to allow cleanup
                cwd=self.workspace_dir
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                return output if output else "Command executed successfully"
            else:
                return f"Command failed with exit code {result.returncode}: {result.stderr.strip()}"
                
        except subprocess.TimeoutExpired:
            return "Command execution timeout"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def _create_directory(self, dir_path: str) -> str:
        """Create directory."""
        try:
            os.makedirs(dir_path, exist_ok=True)
            return f"Directory {dir_path} created successfully"
        except Exception as e:
            return f"Error creating directory: {str(e)}"
    
    def _delete_file(self, file_path: str) -> str:
        """Delete file."""
        try:
            os.remove(file_path)
            return f"File {file_path} deleted successfully"
        except Exception as e:
            return f"Error deleting file: {str(e)}"
    
    def _move_file(self, source_path: str, dest_path: str) -> str:
        """Move file."""
        try:
            shutil.move(source_path, dest_path)
            return f"File moved from {source_path} to {dest_path}"
        except Exception as e:
            return f"Error moving file: {str(e)}"
    
    def _copy_file(self, source_path: str, dest_path: str) -> str:
        """Copy file."""
        try:
            shutil.copy2(source_path, dest_path)
            return f"File copied from {source_path} to {dest_path}"
        except Exception as e:
            return f"Error copying file: {str(e)}"
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available tools in OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and directories in a given path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dir_path": {
                                "type": "string",
                                "description": "Path to the directory to list (default: current directory)",
                                "default": "."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_shell",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory", 
                    "description": "Create a new directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dir_path": {
                                "type": "string",
                                "description": "Path of the directory to create"
                            }
                        },
                        "required": ["dir_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string", 
                                "description": "Path to the file to delete"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_file",
                    "description": "Move or rename a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_path": {
                                "type": "string",
                                "description": "Source file path"
                            },
                            "dest_path": {
                                "type": "string", 
                                "description": "Destination file path"
                            }
                        },
                        "required": ["source_path", "dest_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "copy_file",
                    "description": "Copy a file to another location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_path": {
                                "type": "string",
                                "description": "Source file path"
                            },
                            "dest_path": {
                                "type": "string",
                                "description": "Destination file path"
                            }
                        },
                        "required": ["source_path", "dest_path"]
                    }
                }
            }
        ]
    def _print_workspace_status(self):
        """Print initial workspace directory status for debugging."""
        if not self.workspace_dir:
            print("Workspace directory not initialized")
            return
        
        print(f"\n{'='*50}")
        print(f"Workspace Directory: {self.workspace_dir}")
        print(f"{'='*50}")
        
        try:
            # List all files and directories in workspace
            items = []
            for root, dirs, files in os.walk(self.workspace_dir):
                level = root.replace(self.workspace_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                rel_root = os.path.relpath(root, self.workspace_dir) if root != self.workspace_dir else '.'
                
                if level == 0:
                    print(f"Directory structure:")
                    print(f"{indent}{rel_root}/")
                else:
                    print(f"{indent}{os.path.basename(root)}/")
                
                # Print files in current directory
                sub_indent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        print(f"{sub_indent}{file} ({file_size} bytes)")
                    except:
                        print(f"{sub_indent}{file} (size unknown)")
            
            # Count total items
            total_files = sum([len(files) for r, d, files in os.walk(self.workspace_dir)])
            total_dirs = sum([len(dirs) for r, dirs, f in os.walk(self.workspace_dir)])
            
            print(f"\nSummary: {total_dirs} directories, {total_files} files")
            
        except Exception as e:
            print(f"Error listing workspace: {e}")
        
        print(f"{'='*50}\n")