#!/usr/bin/env python3
"""
Entry Point for Exploration Experiment
"""

import argparse

import gin
import numpy as np

from dair_exploration.file_utils import get_config


## Main Function
@gin.configurable
def main():
    """Main function for online learning loop"""
    # Debug: Remove scientific notation for numpy printing
    np.set_printoptions(suppress=True)

    # Start Input Loop
    def print_help():
        print("\nUsage:\n" + "h - Print Help\n" + "q - Quit\n")

    print_help()
    command_char = " "
    while command_char != "q":
        command_char = input("Command $ ").split(" ")[0]

        if command_char == "h":
            print_help()

    # Quit
    print("Done!")


def main_fn():
    """Entry point"""
    parser = argparse.ArgumentParser(
        prog="run_experiment.py", description="Run the Active Exploration Experiment"
    )
    parser.add_argument(
        "--config_file", default="default.gin", help="GIN config in /config"
    )
    args = parser.parse_args()

    # Parse config file and start
    print(f"Loading Config File: {get_config(args.config_file)}")
    gin.parse_config_file(get_config(args.config_file))
    # Pylint doesn't know about gin
    # pylint: disable-next=no-value-for-parameter
    main()


if __name__ == "__main__":
    main_fn()
