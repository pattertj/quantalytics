---
sidebar_position: 5
---

# Publishing Workflow

Quantalytics ships with a PyPI-friendly layout. Use the following checklist when preparing a release.

## 1. Bump the Version

Update the `version` field in `pyproject.toml` and document the change in `CHANGELOG.md` (optional).

## 2. Run Quality Gates

```bash
pytest
ruff check .
```

## 3. Build Artifacts

```bash
python -m build
```

This produces source and wheel distributions in `dist/`.

## 4. Trigger the Release Workflow

1. Commit the version bump.
2. Create and push a tag that matches the version (e.g., `git tag -a v0.2.0 -m "Release 0.2.0" && git push origin v0.2.0`).

Pushing the tag runs the `Release` GitHub Actions workflow which:

- Validates the tag matches `pyproject.toml`.
- Builds source and wheel distributions.
- Publishes to PyPI using the `PYPI_API_TOKEN` secret.
- Drafts a GitHub release with the build artifacts attached.

> **Secrets**: Configure `PYPI_API_TOKEN` with a PyPI API token that has upload rights for `quantalytics`.

Use `workflow_dispatch` on the Release workflow if you need to rerun the pipeline for a tag. For dry runs, keep the classic commands handy:

```bash
python -m build
twine upload --repository testpypi dist/*
```

## 5. Publish Documentation

```bash
cd docs
npm install
npm run build
```

Deploy the generated `build/` folder to your hosting provider (Netlify, Vercel, GitHub Pages, etc.).
