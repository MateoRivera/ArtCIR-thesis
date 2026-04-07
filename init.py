import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

DEFAULT_PY_VERSION = ">=3.12"
REPLACEABLE_SUFFIXES = {".py", ".toml", ".md", ".yml", ".yaml"}


def ask(prompt: str, default: str | None = None) -> str:
    """Prompt the user for a required string value, with an optional default."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        val = input(f"{prompt}{suffix}: ").strip()
        if val:
            return val
        if default is not None:
            return default
        print("  This field is required.")


def ask_bool(prompt: str, default: bool = True) -> bool:
    """Prompt the user for a yes/no answer."""
    hint = "Y/n" if default else "y/N"
    val = input(f"{prompt} [{hint}]: ").strip().lower()
    return default if not val else val in ("y", "yes")


def query_params() -> dict:
    """Interactively collect project parameters from the user."""
    while True:
        print()
        name = ask("Author name")
        email = ask("Author email")
        project_name = ask("Project name (dist name, e.g. my-project)")
        default_pkg = project_name.lower().replace("-", "_").replace(" ", "_")
        package_name = ask("Package name (import name, e.g. my_package)", default=default_pkg)
        description = ask("Project description")
        py_version = ask("Minimum Python version", default=DEFAULT_PY_VERSION)

        print()
        want_cli = ask_bool("Include CLI entrypoint (cli.py / __main__.py)?", default=True)
        want_tests = ask_bool("Include unit tests (tests/)?", default=True)
        want_docs = ask_bool("Include documentation setup (mkdocs)?", default=True)
        want_precommit = ask_bool("Include pre-commit hooks (.pre-commit-config.yaml)?", default=True)

        print()
        print("Summary:")
        print(f"  Author:      {name} <{email}>")
        print(f"  Project:     {project_name}")
        print(f"  Package:     {package_name}")
        print(f"  Description: {description}")
        print(f"  Python:      {py_version}")
        print(f"  CLI:         {'yes' if want_cli else 'no'}")
        print(f"  Tests:       {'yes' if want_tests else 'no'}")
        print(f"  Docs:        {'yes' if want_docs else 'no'}")
        print(f"  Pre-commit:  {'yes' if want_precommit else 'no'}")

        if ask_bool("Looks good?", default=True):
            return dict(
                name=name,
                email=email,
                project_name=project_name,
                package_name=package_name,
                description=description,
                py_version=py_version,
                want_cli=want_cli,
                want_tests=want_tests,
                want_docs=want_docs,
                want_precommit=want_precommit,
            )
        print("Let's try again.")


def replace_in_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text)


def replace_in_dir(root: Path, replacements: list[tuple[str, str]]) -> None:
    skip = {".git", "__pycache__"}
    for entry in root.rglob("*"):
        if any(part in skip for part in entry.parts):
            continue
        if entry.is_file() and entry.suffix in REPLACEABLE_SUFFIXES:
            replace_in_file(entry, replacements)


def update_license(author_name: str) -> None:
    path = Path("LICENSE")
    if not path.exists():
        return
    text = path.read_text()
    text = text.replace("<year>", str(datetime.now().year)).replace("<name>", author_name)
    path.write_text(text)


def strip_dependency_group(path: Path, group: str) -> None:
    """Remove a named group entry from [dependency-groups] in a toml file."""
    text = path.read_text()
    # matches both single-line and multiline array values
    text = re.sub(rf"^{re.escape(group)}\s*=\s*\[[\s\S]*?\]\n", "", text, flags=re.MULTILINE)
    path.write_text(text)


def update_makefile_sources(path: Path, remove: list[str]) -> None:
    """Remove entries from the `sources = ...` variable in a Makefile."""
    text = path.read_text()

    def rebuild(m: re.Match) -> str:
        parts = [w for w in m.group(2).split() if w not in remove]
        return m.group(1) + " ".join(parts)

    text = re.sub(r"^(sources\s*=\s*)(.+)$", rebuild, text, flags=re.MULTILINE)
    path.write_text(text)


def remove_precommit_from_makefile(path: Path) -> None:
    """Remove the .pre-commit target and the pre-commit install line from the install target."""
    text = path.read_text()
    # remove the .pre-commit phony + target block (up to the next blank line)
    text = re.sub(
        r"\.PHONY: \.pre-commit[^\n]*\n\.pre-commit:[^\n]*\n\t[^\n]*\n\n",
        "\n",
        text,
    )
    # remove pre-commit install line from install target
    text = text.replace("\tuv run pre-commit install --install-hooks\n", "")
    # update install target description
    text = text.replace(
        ".PHONY: install  ## Install all dependencies and pre-commit hooks",
        ".PHONY: install  ## Install all dependencies",
    )
    path.write_text(text)


def main() -> None:
    pyproject = Path("pyproject.toml")
    makefile = Path("Makefile")
    if not pyproject.exists():
        print("No pyproject.toml found, exiting.")
        return

    print("Hi! Let's set up your project.")
    try:
        p = query_params()
    except KeyboardInterrupt:
        print("\nExiting.")
        return

    # order matters: replace the more specific (longer) strings first
    # to avoid partial matches (e.g. "project-name" before "project")
    replacements: list[tuple[str, str]] = [
        ("project-name", p["project_name"]),   # dist name  (dashes, in pyproject/cli/readme)
        ("project_name", p["package_name"]),   # import name (underscores, in .py files)
        ("author name", p["name"]),
        ("author@email.com", p["email"]),
        ("Project description", p["description"]),
        (">=3.12", p["py_version"]),
    ]

    print("\nUpdating license...")
    update_license(p["name"])

    print("Renaming package directory...")
    pkg_dir = Path("src") / p["package_name"]
    Path("src/project_name").rename(pkg_dir)

    print("Replacing placeholders...")
    dirs_to_process = [Path("src"), Path("tools")]
    if p["want_tests"]:
        dirs_to_process.append(Path("tests"))
    for d in dirs_to_process:
        if d.exists():
            replace_in_dir(d, replacements)
    for f in (pyproject, Path("README.md"), Path("mkdocs.yml")):
        if f.exists():
            replace_in_file(f, replacements)

    if not p["want_cli"]:
        print("Removing CLI files...")
        for fname in ("cli.py", "__main__.py"):
            (pkg_dir / fname).unlink(missing_ok=True)

    if not p["want_tests"]:
        print("Removing tests/...")
        shutil.rmtree("tests", ignore_errors=True)
        strip_dependency_group(pyproject, "test")
        if makefile.exists():
            update_makefile_sources(makefile, ["tests"])

    if not p["want_docs"]:
        print("Removing docs/...")
        shutil.rmtree("docs", ignore_errors=True)
        Path("mkdocs.yml").unlink(missing_ok=True)
        strip_dependency_group(pyproject, "docs")

    if not p["want_precommit"]:
        print("Removing pre-commit config...")
        Path(".pre-commit-config.yaml").unlink(missing_ok=True)
        if makefile.exists():
            remove_precommit_from_makefile(makefile)

    print("\nInitializing environment...")
    try:
        result = subprocess.run(["uv", "sync", "--all-groups"])
        if result.returncode != 0:
            print("  uv sync failed — run it manually when ready.")
    except FileNotFoundError:
        print("  uv not found — install it, then run: uv sync --all-groups")

    print("\nYour project is ready. Deleting myself, farewell!")
    Path(__file__).unlink()


if __name__ == "__main__":
    main()
