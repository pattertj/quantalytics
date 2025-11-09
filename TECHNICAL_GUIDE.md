# Quantalytics Technical Guide

Audience: developers and AI agents contributing to Quantalytics. This document explains how the codebase is organized, how to extend the API, and what quality bars (naming, testing, documentation) every change must satisfy. It intentionally lives outside the Docusaurus end-user docs.

## Architectural Overview

The repository is a single Python package (`quantalytics`) that exposes *namespaces*—not individual functions—from the root import. Consumers are expected to write:

```python
import quantalytics as qa

qa.metrics.sharpe_ratio(...)
qa.stats.skewness(...)
qa.charts.cumulative_returns_chart(...)
qa.reports.render_basic_tearsheet(...)
```

The top-level package simply forwards these modules via `__all__ = ["metrics", "stats", "charts", "reports"]`. Keep the root namespace lean; anything new should reside in a submodule unless it is unavoidable package metadata.

### Package Layout

| Location | Responsibility | Notes |
| --- | --- | --- |
| `quantalytics/metrics` | Risk/return ratios and summaries (`PerformanceMetrics`, Sharpe, Sortino, Calmar, drawdowns). | Depends on pandas/NumPy; accepts iterables or Series. |
| `quantalytics/stats` | Pure descriptive statistics (skew, kurtosis, total return, CAGR). | Avoid annualized risk-adjusted measures here. |
| `quantalytics/charts` | Plotly-based visualization helpers. | Input validation and layout defaults live close to the chart functions. |
| `quantalytics/reporting` | Tear sheet assembly, HTML templates. | Templates stored under `quantalytics/reporting/templates`. |
| `quantalytics/utils` | Cross-cutting helpers (e.g., timeseries utilities). | Keep tiny, reusable primitives here. |

`tests/` mirrors these domains (`test_metrics.py`, `test_stats.py`, …). When adding a new module, add a matching test file.

## Naming & API Expectations

1. **Clarity over brevity inside namespaces.** Functions should describe *what* they return: use `sharpe_ratio`, `total_return`, `rolling_volatility_chart`. The only short names live on dataclasses or result objects (e.g., `PerformanceMetrics.sharpe`).
2. **Avoid collisions at the root.** Because consumers interact via `qa.metrics` and `qa.stats`, do not re-export individual helpers at `quantalytics/__init__.py`.
3. **Document every public helper.** Include a docstring with a one-line summary plus any non-obvious parameter behavior (annualization factors, units, return types). Type hints are required.
4. **Shared helpers stay private.** Prefix internal helpers with `_` (`_to_series`, `_annualization_factor`) and keep them in the module where they are used unless multiple namespaces need them—in which case, move them to `quantalytics.utils` and add tests.

### When to add new stats vs metrics

- **Stats (`qa.stats`)**: basic descriptive analytics that do not require a benchmark or risk model (skewness, kurtosis, CAGR, cumulative growth).
- **Metrics (`qa.metrics`)**: ratios or measurements that relate return to risk, drawdown, or targets (Sharpe, Sortino, max drawdown, downside deviation).

If unsure, ask whether the function’s output could be computed by a statistics package without portfolio context; if yes, it probably belongs in `stats`.

## Testing Requirements

- **Command**: all changes must pass `uv run pytest --cov=quantalytics --cov-report=term-missing`. When the command cannot execute (e.g., uv not installed), note it in your PR/commit message and run the suite locally before merging.
- **Coverage**: pair every new public helper with at least one test that asserts numerical correctness (use `pytest.approx` when comparing floats).
- **Determinism**: Seed any RNG usage (see `tests/test_metrics.py`’s `np.random.default_rng(42)` pattern).
- **Negative paths**: if a helper raises for invalid input, add an explicit test (e.g., non-numeric data passed to `_to_series` equivalents).
- **Pre-commit hooks**: run `pre-commit install` once, then `pre-commit run --all-files` to populate caches. Hooks will run automatically on each commit; CI assumes they’ve been executed locally.

## Documentation Workflow

1. **User-facing docs** live under `docs/docs/` (Docusaurus). Update those when the public API changes.
2. **Technical notes** (like this file) stay outside `docs/docs/` to avoid being published on the public site.
3. Whenever you add a new helper or pattern, update both the relevant Docusaurus guide and this file’s corresponding section (package layout table, naming guidance, etc.).

## Contribution Checklist

Before submitting a PR or letting an AI agent finish a task:

1. Ensure naming follows the clarity-first rule and the helper sits in the correct namespace.
2. Run formatting/linting if configured (currently `ruff check .` in Development instructions).
3. Run `uv run pytest --cov=quantalytics --cov-report=term-missing` (or document why it could not run).
4. Run `pre-commit run --all-files` until the hook set passes cleanly.
4. Update README quickstart/examples if the change affects core workflows.
5. Update the Docusaurus docs and this technical guide when public behavior shifts.

Following these steps keeps the package predictable for both humans and agents automating strategy analytics with Quantalytics.
