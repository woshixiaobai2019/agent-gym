"""Main script for running Python Interpreter tasks."""

import argparse
from pathlib import Path
from agent_gym.runners.python_runner import PythonRunner


def main():
    """Main function for Python interpreter testing."""
    parser = argparse.ArgumentParser(description="Run Python Interpreter Tasks")
    parser.add_argument("--data-file", type=str, default="agent_gym/data/py.json",
                       help="Path to task data file")
    parser.add_argument("--task-id", type=int, default=None,
                       help="Specific task ID to run (if not provided, runs all)")
    parser.add_argument("--agent-url", type=str, default="http://localhost:7996",
                       help="Agent API base URL")
    parser.add_argument("--sandbox-url", type=str, default="http://localhost:8080",
                       help="Sandbox API base URL")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Request timeout in seconds")
    parser.add_argument("--max-turns", type=int, default=10,
                       help="Maximum turns per task")
    parser.add_argument("--log-dir", type=str, default="logs/python",
                       help="Directory to save logs")
    parser.add_argument("--quiet", action="store_true",
                       help="Run in quiet mode (less verbose output)")
    
    args = parser.parse_args()
    
    # Validate data file exists
    if not Path(args.data_file).exists():
        print(f"❌ Data file not found: {args.data_file}")
        return
    
    # Initialize runner
    runner = PythonRunner(
        data_file=args.data_file,
        agent_base_url=args.agent_url,
        sandbox_url=args.sandbox_url,
        request_timeout=args.timeout,
        log_dir=args.log_dir
    )
    
    verbose = not args.quiet
    
    try:
        if args.task_id is not None:
            # Run specific task
            print(f"🎯 Running specific task: {args.task_id}")
            result = runner.run_single_task(args.task_id, verbose=verbose, max_turns=args.max_turns)
            print(f"\n✅ Task completed. Success: {result.get('success', False)}")
        else:
            # Run all tasks
            print(f"🚀 Running all tasks from: {args.data_file}")
            summary = runner.run_all_tasks(verbose=verbose, max_turns=args.max_turns)
            print(f"\n🏁 All tasks completed. Success rate: {summary['success_rate']:.1%}")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")


if __name__ == "__main__":
    main()