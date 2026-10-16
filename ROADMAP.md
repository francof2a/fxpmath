# Fxpmath Roadmap

This roadmap communicates high-level priorities and milestones for upcoming releases.
It is intended for contributors and users who want visibility into project direction.

## Current Focus: v0.5.0

`v0.5.0` is the next major release and is focused on three themes:
- New capabilities (NumPy interoperability and feature expansion)
- Better internal structure and maintainability
- Measurable runtime performance improvements

## Milestones for v0.5.0

### M1 - Stability and Compatibility Baseline
- Finalize NumPy compatibility behavior for supported versions (`1.26.x` and latest `2.x`)
- Resolve remaining correctness edge cases in arithmetic and dispatch paths
- Add missing regression coverage for known high-risk scenarios
- Align core NumPy protocol hooks with modern expectations (`__array__`, `__array_wrap__`, `out` behavior)

**Exit criteria**
- Full test suite green on required compatibility matrix
- No known critical correctness regressions open for `v0.5.0`

### M2 - CI and Release Guardrails
- Enforce required CI checks for lint + tests across supported Python and OS versions
- Add required NumPy version matrix jobs
- Add package build/install verification in CI
- Add release checklist gate for compatibility and quality reports

**Exit criteria**
- Required checks are stable and blocking for `v0.5.0` branches/PRs
- Release candidate can be validated end-to-end from CI artifacts

### M3 - Core Refactor and Maintainability
- Reduce complexity in core modules (starting with `objects.py`) via internal decomposition
- Replace broad exception handling with typed/explicit error paths where practical
- Improve test organization and naming consistency
- Establish lint/style ratchet to prevent maintenance debt growth

**Exit criteria**
- Core refactor lands without behavior regressions
- Maintainability metrics trend positively (complexity/lint debt does not increase)

### M4 - Feature Expansion
- Expand NumPy dispatch coverage for high-value operations (`vecdot`, `matvec`, `vecmat`, norms)
- Support matrix multiplication (`@` operator and `np.matmul`) with exact fixed-point accumulation
- Improve behavior consistency for `out`/`out_like` and dispatch kwargs
- Add a fixed-width hardware modeling profile (config preset where operation results keep the operand size and overflow is handled by the configured behavior), without changing current defaults
- Document feature behavior and compatibility caveats, including a guide on modeling fixed-width hardware (`op_sizing`, overflow modes, flags)

**Exit criteria**
- Newly added dispatch features have tests and documentation
- Public behavior is consistent with documented expectations

### M5 - Performance Milestone
- Add repeatable benchmark suite for arithmetic hot paths (`add`, `mul`, `truediv`, `pow`)
- Record baseline metrics for `v0.5.0` development
- Implement targeted optimizations and track deltas
- Cover the slow paths reported by users: `Fxp` creation from large arrays (#46), per-element access and assignment in loops (#74), and `pow` (#65)
- Publish comparative benchmarks against plain NumPy and other fixed-point libraries (e.g. APyTypes, numfi)
- Add regression guardrails for performance in CI or release validation

**Exit criteria**
- Benchmarks are reproducible and versioned
- Targeted hot paths show measurable, validated improvements
- Comparative results are published in the documentation

### M6 - Citability and Ecosystem Visibility
- Add a `CITATION.cff` file and archive releases on Zenodo to get a DOI
- Add a citation section to the README and documentation
- Add a comparison page with other Python fixed-point libraries (features, semantics, performance)
- Request corrections to third-party comparison tables that describe fxpmath inaccurately (e.g. implementation language, `@` support once available)

**Exit criteria**
- `v0.5.0` release has a citable DOI
- Comparison page is published and linked from the README

## Release Readiness (v0.5.0)

`v0.5.0` is considered release-ready when:
- Stability milestones (M1, M2) are complete
- Refactor changes (M3) are merged with no major regressions
- Selected feature scope (M4) is complete with tests/docs
- Performance milestone (M5) provides baseline + improvement evidence
- Citability milestone (M6) provides a DOI for the release

## Looking Ahead (Post v0.5.0)

After `v0.5.0`, roadmap priorities are expected to include:
- Broader API/NumPy feature coverage
- Additional performance optimization passes
- Compiled engine for extended precision (>64-bit words)
- Short software paper (e.g. JOSS) describing the library
- Documentation depth and developer experience improvements toward `v1.0`

## Contribution Notes

- Roadmap items may be adjusted based on bug reports, contributor bandwidth, and ecosystem changes.
- If you want to help, open or comment on an issue and reference the milestone (`M1`-`M6`).
- Small, focused pull requests aligned to roadmap milestones are preferred.
