from project_name.core import greet


def main() -> None:
    # example of using the library directly, without going through the CLI
    result = greet("world")
    print(result)


if __name__ == "__main__":
    main()
