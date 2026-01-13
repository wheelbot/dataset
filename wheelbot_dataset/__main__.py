"""
wheelbot-dataset package main entry point.

Allows running the package as a module:
    python -m wheelbot_dataset record vel
    python -m wheelbot_dataset example
"""

import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m wheelbot_dataset <command>")
        print("Commands:")
        print("  record       - Run the recording CLI (use 'record --help' for options)")
        print("  consolidate  - Consolidate or analyze datasets (use 'consolidate --help' for options)")
        print("  example      - Run the example usage script")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "record":
        # Remove the 'record' command from argv so fire sees the subcommand
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from wheelbot_dataset.record import fire
        import wheelbot_dataset.record as record_module
        fire.Fire({
            "vel": record_module.vel,
            "roll": record_module.roll,
            "roll_max": record_module.roll_max,
            "pitch": record_module.pitch,
            "yaw": record_module.yaw,
            "velrollpitch": record_module.velrollpitch,
            "velrollpitch2": record_module.velrollpitch2,
            "lin": record_module.lin,
            "linwithlean": record_module.linwithlean,
            "lin2": record_module.lin2,
        })
    elif command == "consolidate":
        # Remove the 'consolidate' command from argv so fire sees the subcommand
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from wheelbot_dataset.consolidate import fire
        import wheelbot_dataset.consolidate as consolidate_module
        fire.Fire({
            "consolidate": consolidate_module.consolidate,
            "statistics": consolidate_module.statistics,
        })
    elif command == "example":
        from wheelbot_dataset.example_usage import example_usage
        example_usage()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: record, consolidate, example")
        sys.exit(1)


if __name__ == "__main__":
    main()
