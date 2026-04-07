# python ML template

A simple template to bootstrap Python packages with [uv](https://docs.astral.sh/uv/).

## Features

- **uv-native** — environment management, dependency resolution, and versioning in one tool.
- **One config file** — `pyproject.toml` handles everything: dependencies, tools, versioning.
- **Ruff** — single tool for linting and formatting (replaces black, isort, flake8).
- **hatchling** — lightweight, modern build backend.

## Getting Started

1. Click "Use this template" and clone your new repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Run the initialization script and follow the prompts:

   ```bash
   python init.py
   ```

4. Install all dependencies:

   ```bash
   uv sync --all-groups
   ```

5. You're good to go. Run `make help` to see all available commands.

> **Note**
>
> `init.py` is self-contained and will delete itself once completed.
> It is safe to delete and edit the files manually instead.

## Common Commands

| Command        | Description                                  |
|----------------|----------------------------------------------|
| `make sync`    | Install all dependency groups                |
| `make fmt`     | Format and auto-fix code with ruff           |
| `make lint`    | Check style and lint (no changes)            |
| `make test`    | Run tests with coverage report               |
| `make clean`   | Remove build artifacts and cache files       |
| `make release` | Bump version, tag, and push a new release    |

## VS Code

To format and sort imports on save, add this to your `settings.json`:

```json
{
    "[python]": {
        "diffEditor.ignoreTrimWhitespace": false,
        "editor.wordBasedSuggestions": "off",
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        }
    }
}
```
