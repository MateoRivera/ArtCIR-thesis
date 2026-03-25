import argparse

from artcir_thesis.core import greet


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="artcir-thesis CLI")
    parser.add_argument("name", help="name to greet")
    args = parser.parse_args()
    print(greet(args.name))
