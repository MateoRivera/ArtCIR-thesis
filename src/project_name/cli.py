import argparse

from project_name.core import greet


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="project-name CLI")
    parser.add_argument("name", help="name to greet")
    args = parser.parse_args()
    print(greet(args.name))
