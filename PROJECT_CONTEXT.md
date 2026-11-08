# House Edge challenge context
The active task is House Edge slot forecasting, per the user's latest request and attached House Edge -- TAMU Datathon 2026 implementation plan. Earlier image-inpainting instructions concern a different project. No existing PROJECT_CONTEXT.md was present.

Objective: predict probabilities for payouts 0,1,2,5,20 using released histories strictly before cutoff. Use the unchanged official scorer; metadata sets main=60 and adaptation=40. The participant PDF has older 80/20 and upload instructions; use this kit's RULES.md and metadata for local implementation.

Current increment: preserve pooled_recent fallback; implement/test safe features; compare a bounded CPU model/transition/blend candidate set; package selected predictions. No GPU, downloads, uploads, or neural architecture work.

Validation: deterministic condition-stratified episode split from the kit. Select on tuning only; inspect selected model and fallback on audit once. The starter already reports full-validation baseline scores, so audit is reserved from new candidate selection, not a completely unseen dataset. Further tuning after inspecting audit needs a new evaluation protocol.

Historical augmentation excludes signal, type, cutoff-as-feature and machine index because historical metadata availability/ordering is not established. All model features derive from observed histories. Missing ticks have explicit indicators. Whole episodes remain together.

Original organizer files and data must remain unchanged. Check SHA256.json. Outputs are local preliminary artifacts, not submitted entries. Fresh-folder local reproduction does not prove Linux organizer-runtime reproduction.

Second increment complete: portable_01 is the recommended matching CSV/ZIP; standard-library inference verified on Windows and Ubuntu WSL2. Frozen score unchanged. Tuning-only diagnostics and 3-fold training-only protocol saved in runs/diagnostics_01. Original audit must not be reused for tuning. Next bounded experiment should use that train-only protocol.

Third increment complete: runs/train_cv_01 compared incumbent, regularized, and higher_capacity settings using only training folds. Incumbent won (observed Brier 0.544956); no promotion, validation/test not read. portable_01 remains recommended. Four new CV tests passed (14 total). Next candidate should target machine-specific temporal features, not repeat this capacity sweep.

Fourth increment complete: runs/temporal_cv_01 compared predeclared machine-local temporal candidates and fixed blends on the same training-only episode folds. binary_short_w1.0 passed the promotion rule on all three folds and was then evaluated once on public validation, scoring 39.324354/60 versus the incumbent's 32.107314/60. The promoted model's test CSV and standard-library package are in submissions/temporal_02; fresh extracted-package reproduction differs by at most 8.88e-16. All 20 unit tests and both test-format checks passed. This is still a public-validation result, not a hidden score. Preserve portable_01 as the rollback entry.

Fifth increment complete: solution/structural.py and solution/fit_structural.py replace frequency models with Bayesian inference over the recovered simulator rules. Cross-validated validation score 58.91/60 (R 0.018, bootstrap 58.60-59.18), against 39.32 for temporal_02. Recommended pair: submissions/structural_final. It needs NumPy, pinned in its requirements.txt, and reproduced on Windows and Ubuntu WSL2 (1.63e-13). 32 unit tests passed; all organizer hashes match. Nothing has been uploaded. Round two steps are in ROUND_TWO_RUNBOOK.md; the fit relearns all rules from the new kit. Details in SOLUTION.md.

Requirements audit complete: recommended pair is now submissions/structural_final_v2 (same CSV as structural_final, ZIP has template README and training code). Conservative audit estimate about 58.75/60. 34 tests pass; Python 3.13 runner simulation matches to 1.63e-13. Details at the end of SOLUTION.md.

Round-two stress test complete: rules are now learned truth tables over 1-3 lags, plus a catch-all sticky-regime condition and a family_check in report.json. Stress results and the table are at the end of SOLUTION.md and in ROUND_TWO_RUNBOOK.md. Main score unchanged (58.913 cross-validated); main entry unchanged (structural_final_v2). Round-two fit takes about 10 minutes on an idle machine.
