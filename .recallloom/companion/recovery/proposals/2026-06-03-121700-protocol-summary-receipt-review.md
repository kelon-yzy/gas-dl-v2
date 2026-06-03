# Recovery proposal: protocol summary receipt mismatch

## 来源摘要

`recallloom.py write --type current-state` updated `rolling_summary.md` to revision 35 and `state.json` to workspace revision 68, but receipt finalization failed with `post_hash_mismatch`.

## 来源类型与可信级别

- Source type: local helper diagnostics.
- Confidence: high for structural validity, low for receipt-backed provenance.

- `validate_context.py --json` reports structural validity with zero errors and one legacy optional metadata warning.
- `validate_context.py --require-provenance --changed-only` reports receipt-store evidence mismatch/missing, so mutating writes are blocked until recovery review is recorded.
- `preflight_context_check.py --json` reports `summary_stale=false`, `workspace_revision=68`, `rolling_summary_revision=35`, but `provenance_state=inconsistent_or_tampered_evidence`.

## 候选当前状态事实

- `rolling_summary.md` revision 35 reflects the current project state after the long-sequence protocol implementation.
- `state.json` workspace revision is 68.
- Managed file structural validation is valid.

## 候选里程碑事件

- Long-sequence protocol S0-S5 implementation reached a code/test/docs checkpoint.
- `python -m pytest` passed with 170 tests.

## 候选判断反转

- The sidecar should not be treated as receipt-backed for the failed summary write.
- The sidecar can be treated as a reviewed structural baseline after recovery review.

## 候选下一步变化

- After recovery, rerun preflight before any further daily-log append.
- Use helper writes only.

## 与当前 sidecar 的冲突

- Receipt store evidence is inconsistent with current state after `post_hash_mismatch`.
- Structural files are readable and current, but provenance must be downgraded/reviewed.

## 建议提升动作

Treat the current structurally valid managed files at workspace revision 68 as a reviewed imported baseline for this local sidecar. Do not claim receipt-backed provenance for the failed write. Preserve the current rolling summary content because it accurately records the long-sequence protocol implementation state and test result.

## 审阅结论

Pending reviewer approval. Proposed action is accept after explicit recovery review.
