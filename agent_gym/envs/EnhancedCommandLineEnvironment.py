"""Enhanced Command Line Environment for Agent Gym."""

import os
import shutil
import json
import logging
import tempfile
from typing import Dict, List, Any, Tuple, Optional

from .CommandLineEnvironment import CommandLineEnvironment
from ..core.types import ObservationType, ActionType, ToolsType


class EnhancedCommandLineEnvironment(CommandLineEnvironment):
    """Enhanced command line environment that supports extended data format and features."""
    
    def __init__(self, data_file: str = None, task_id: int = 0):
        """Initialize the enhanced command line environment.
        
        Args:
            data_file: Path to JSON file containing tasks (supports both old and new formats)
            task_id: Index of the task to load from data file
        """
        # Set up logging first
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize enhanced properties
        self.allowed_commands: Optional[List[str]] = None
        self.task_level: Optional[str] = None
        self.format_example: Optional[str] = None
        self.reference_answer: Optional[str] = None
        
        # Call parent constructor
        super().__init__(data_file, task_id)
    
    def _load_task_data(self):
        """Load task data from file, supporting both old and new formats."""
        if not self.data_file:
            return
            
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            
            if 0 <= self.task_id < len(tasks):
                task_data = tasks[self.task_id]
                self.current_task = task_data
                
                # Detect and handle new format
                if self._is_enhanced_format(task_data):
                    self._load_enhanced_task_data(task_data)
                else:
                    self._load_legacy_task_data(task_data)
                    
        except Exception as e:
            self.logger.error(f"Failed to load task data: {e}")
            raise EnvironmentError(f"Failed to load task data: {e}")
    
    def _is_enhanced_format(self, task_data: Dict[str, Any]) -> bool:
        """Detect if the task data uses the enhanced format."""
        # Enhanced format has 'task' field and may have 'create_env'
        return 'task' in task_data and isinstance(task_data['task'], dict)
    
    def _load_enhanced_task_data(self, task_data: Dict[str, Any]):
        """Load enhanced format task data."""
        self.logger.info("Loading enhanced format task data")
        
        task = task_data['task']
        
        # Extract enhanced task properties
        self.task_level = task.get('level')
        self.format_example = task.get('format_example')
        self.allowed_commands = task.get('allowed_commands', [])
        self.reference_answer = task.get('reference_answer')
        
        # Log task details
        self.logger.info(f"Task level: {self.task_level}")
        self.logger.info(f"Allowed commands: {len(self.allowed_commands) if self.allowed_commands else 'All'}")
        
    def _load_legacy_task_data(self, task_data: Dict[str, Any]):
        """Load legacy format task data for backward compatibility."""
        self.logger.info("Loading legacy format task data")
        
        # Reset enhanced properties for legacy format
        self.allowed_commands = None
        self.task_level = None
        self.format_example = None
        self.reference_answer = None
    
    def reset(self) -> Tuple[ObservationType, ToolsType]:
        """Reset the environment and set up workspace with enhanced features."""
        # Clean up previous workspace
        self.cleanup()
        
        # Create new temporary workspace directory
        self.workspace_dir = tempfile.mkdtemp(prefix="agent_gym_enhanced_")
        self.logger.info(f"Created workspace: {self.workspace_dir}")
        
        if not self.current_task:
            return "Hello! How can I help you?", self.tools
        
        try:
            # Setup main environment
            if self._is_enhanced_format(self.current_task):
                # Enhanced format setup
                if 'create_env' in self.current_task:
                    self._setup_environment(self.current_task['create_env'])
                    self.logger.info("Main environment created successfully")
                
                # Setup noise environment if specified
                if 'create_noise_env_appendix' in self.current_task:
                    self._setup_noise_environment(self.current_task['create_noise_env_appendix'])
                    self.logger.info("Noise environment created successfully")
                
                # Get query from task.question
                query = self.current_task['task'].get('question', 'No question provided')
                
            else:
                # Legacy format setup
                if 'env' in self.current_task:
                    self._setup_environment(self.current_task['env'])
                
                query = self.current_task.get('query', 'Hello! How can I help you?')
            
            # Print enhanced workspace status
            self._print_enhanced_workspace_status()
            
            return query, self.tools
            
        except Exception as e:
            self.logger.error(f"Error during environment setup: {e}")
            return f"Error setting up environment: {str(e)}", self.tools
    
    def _setup_noise_environment(self, noise_script: str):
        """Set up noise/distraction environment using the provided script."""
        if not noise_script or not noise_script.strip():
            return
            
        original_cwd = os.getcwd()
        try:
            os.chdir(self.workspace_dir)
            self.logger.info("Executing noise environment script")
            
            # Execute the noise environment setup script
            exec(noise_script, {"os": os, "__workspace__": self.workspace_dir})
            
        except Exception as e:
            self.logger.error(f"Error setting up noise environment: {e}")
            raise
        finally:
            os.chdir(original_cwd)
    
    def _execute_shell(self, command: str) -> str:
        """Execute shell command with command filtering based on allowed_commands."""
        # Check command filtering for enhanced format
        if (self._is_enhanced_format(self.current_task) and 
            self.allowed_commands is not None and 
            len(self.allowed_commands) > 0):
            
            # Extract the base command (first word)
            base_command = command.strip().split()[0] if command.strip() else ""
            
            if base_command not in self.allowed_commands:
                self.logger.warning(f"Command '{base_command}' not in allowed commands: {self.allowed_commands}")
                return f"Error: Command '{base_command}' is not allowed. Allowed commands: {', '.join(self.allowed_commands)}"
        
        # Use parent implementation for actual execution
        return super()._execute_shell(command)
    
    def _verify_task_completion(self) -> bool:
        """Verify task completion using enhanced verification method."""
        if not self.current_task:
            return False
        
        try:
            original_cwd = os.getcwd()
            os.chdir(self.workspace_dir)
            
            if self._is_enhanced_format(self.current_task):
                # Enhanced format verification
                task = self.current_task['task']
                if 'verify_answer' in task:
                    return self._execute_enhanced_verification(task['verify_answer'])
                else:
                    self.logger.warning("No verify_answer field found in enhanced format")
                    return False
            else:
                # Legacy format verification
                if 'verify' in self.current_task:
                    return self._execute_legacy_verification(self.current_task['verify'])
                else:
                    self.logger.warning("No verification method found")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Verification error: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def _execute_enhanced_verification(self, verify_code: str) -> bool:
        """Execute enhanced verification code."""
        try:
            local_vars = {
                "os": os, 
                "__workspace__": self.workspace_dir,
                "success": False
            }
            
            exec(verify_code, local_vars)
            result = local_vars.get("success", False)
            
            self.logger.info(f"Enhanced verification result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced verification error: {e}")
            return False
    
    def _execute_legacy_verification(self, verify_code: str) -> bool:
        """Execute legacy verification code for backward compatibility."""
        try:
            local_vars = {
                "os": os, 
                "__workspace__": self.workspace_dir,
                "success": False
            }
            
            exec(verify_code, local_vars)
            result = local_vars.get("success", False)
            
            self.logger.info(f"Legacy verification result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Legacy verification error: {e}")
            return False
    
    def _print_enhanced_workspace_status(self):
        """Print enhanced workspace directory status with additional information."""
        if not self.workspace_dir:
            print("Workspace directory not initialized")
            return
        
        print(f"\n{'='*60}")
        print(f"ENHANCED WORKSPACE STATUS")
        print(f"{'='*60}")
        print(f"Workspace Directory: {self.workspace_dir}")
        
        # Print task information if available
        if self.current_task and self._is_enhanced_format(self.current_task):
            task = self.current_task['task']
            print(f"Task Level: {task.get('level', 'Unknown')}")
            print(f"Question: {task.get('question', 'No question')[:100]}{'...' if len(task.get('question', '')) > 100 else ''}")
            
            if self.format_example:
                print(f"Expected Format: {self.format_example}")
            
            if self.allowed_commands is not None:
                if len(self.allowed_commands) == 0:
                    print("Allowed Commands: All commands allowed")
                else:
                    print(f"Allowed Commands: {', '.join(self.allowed_commands)}")
            
            if self.reference_answer:
                print(f"Reference Answer: {self.reference_answer[:100]}{'...' if len(self.reference_answer) > 100 else ''}")
        
        print(f"{'='*60}")
        
        # Show directory structure
        try:
            total_files = 0
            total_dirs = 0
            
            print("Directory Structure:")
            for root, dirs, files in os.walk(self.workspace_dir):
                level = root.replace(self.workspace_dir, '').count(os.sep)
                indent = '  ' * level
                rel_path = os.path.relpath(root, self.workspace_dir) if root != self.workspace_dir else '.'
                
                if level == 0:
                    print(f"{indent}{rel_path}/")
                else:
                    print(f"{indent}{os.path.basename(root)}/")
                
                # Show files
                sub_indent = '  ' * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        print(f"{sub_indent}{file} ({file_size} bytes)")
                        total_files += 1
                    except:
                        print(f"{sub_indent}{file} (size unknown)")
                        total_files += 1
                
                total_dirs += len(dirs)
            
            print(f"\nSummary: {total_dirs} directories, {total_files} files")
            
        except Exception as e:
            self.logger.error(f"Error listing workspace: {e}")
            print(f"Error listing workspace: {e}")
        
        print(f"{'='*60}\n")
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get detailed information about the current task."""
        if not self.current_task:
            return {}
        
        info = {
            "format": "enhanced" if self._is_enhanced_format(self.current_task) else "legacy",
            "workspace_dir": self.workspace_dir
        }
        
        if self._is_enhanced_format(self.current_task):
            task = self.current_task['task']
            info.update({
                "level": task.get('level'),
                "question": task.get('question'),
                "format_example": task.get('format_example'),
                "allowed_commands": task.get('allowed_commands'),
                "reference_answer": task.get('reference_answer'),
                "has_noise_env": 'create_noise_env_appendix' in self.current_task
            })
        else:
            info.update({
                "query": self.current_task.get('query'),
                "has_env": 'env' in self.current_task,
                "has_verify": 'verify' in self.current_task
            })
        
        return info