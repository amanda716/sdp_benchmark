import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Process task type and config file.")
    parser.add_argument("--task", type=str, required=True, help="Type of the task to perform.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()

    task_type = args.task
    config_path = args.config

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    print(f"Task Type: {task_type}")
    print(f"Config: {config}")
    print("Hello, World!")

if __name__ == "__main__":
    main()    
