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

## 4. Upload to PyPI

```bash
twine upload dist/*
```

Use `--repository testpypi` when rehearsing a release.

## 5. Publish Documentation

```bash
cd docs
npm install
npm run build
```

Deploy the generated `build/` folder to your hosting provider (Netlify, Vercel, GitHub Pages, etc.).
