# Contributing to aether-ai

Thanks for your interest in improving the Python SDK for
[Aether](https://aetherdb.ai)! Bug reports, fixes, docs, and features are all welcome.

## Getting started

```bash
git clone https://github.com/quintessence-group/aether-sdk-python.git
cd aether-sdk-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Development workflow

1. Fork the repo and create a topic branch off `main`.
2. Make a focused change, covered by tests.
3. Run the test suite (below) — everything should pass.
4. Open a pull request describing the change and its motivation.

### Test

```bash
pytest
```

The package ships type information (`py.typed`); please keep public APIs fully type-hinted
and follow [PEP 8](https://peps.python.org/pep-0008/).

## Guidelines

- Add or update tests for any behavior change.
- Update `README.md` for any user-facing change.
- Keep public API changes backward-compatible where possible; call out breaking changes
  clearly in the PR.

## Reporting issues

- **Bugs / features:** open a GitHub issue.
- **Security vulnerabilities:** follow [SECURITY.md](SECURITY.md) — please do not file a
  public issue.

## License

By contributing, you agree that your contributions will be licensed under the project's
[MIT License](LICENSE).
