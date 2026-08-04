# Development workflow

Use the repository's locked `uv` environment for all Python work:

```sh
uv sync
uv run pytest tests/ -q
uv run ruff check .
```

Run the full pipeline with `uv run python run.py`. Do not use bare `python`,
`pip`, a manually activated virtual environment, or add project-local platform
environment files. Update dependencies with `uv add`/`uv remove`, and commit
both `pyproject.toml` and `uv.lock`. The `.venv` directory is local and must
remain ignored.

The committed option-chain snapshot and generated figures are reproducibility
artefacts. Do not refresh or regenerate them during unrelated changes.
