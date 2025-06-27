"""Main script for running NLP Environment tasks."""

import argparse
from pathlib import Path
from agent_gym.runners.nlp_runner import NLPRunner


def main():
    """Main function for NLP environment testing."""
    parser = argparse.ArgumentParser(description="Run NLP Environment Tasks")
    parser.add_argument("--data-file", type=str, default="agent_gym/data/xiaohongshu.json",
                       help="Path to task data file")
    parser.add_argument("--task-id", type=int, default=None,
                       help="Specific task ID to run (if not provided, runs all)")
    parser.add_argument("--agent-url", type=str, default="http://localhost:7996",
                       help="Agent API base URL")
    parser.add_argument("--env-llm-url", type=str, default="http://localhost:8000",
                       help="Environment LLM API base URL")
    parser.add_argument("--env-llm-key", type=str, default="dummy",
                       help="Environment LLM API key")
    parser.add_argument("--max-turns", type=int, default=20,
                       help="Maximum conversation turns")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Request timeout in seconds")
    parser.add_argument("--no-stream", action="store_true",
                       help="Disable streaming responses")
    parser.add_argument("--log-dir", type=str, default="logs/nlp",
                       help="Directory to save logs")
    parser.add_argument("--quiet", action="store_true",
                       help="Run in quiet mode (less verbose output)")
    
    args = parser.parse_args()
    
    # Validate data file exists
    if not Path(args.data_file).exists():
        print(f"❌ Data file not found: {args.data_file}")
        return
    
    # Initialize runner
    runner = NLPRunner(
        data_file=args.data_file,
        agent_base_url=args.agent_url,
        env_llm_base_url=args.env_llm_url,
        env_llm_api_key=args.env_llm_key,
        max_turns=args.max_turns,
        stream=not args.no_stream,
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