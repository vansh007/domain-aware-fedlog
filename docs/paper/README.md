# Paper source (LaTeX)

`main.tex` is the **complete IEEE-format paper** (Abstract → Conclusion + References), with real
related-work comparisons and citations in `references.bib`. Every number traces to a file in
`../../results/`; provenance and exact commands are in `docs/RESEARCH_LOG.md` and
`docs/paper/HANDOFF.md`.

## Status
- Full paper written and self-verified: all numbers match the CSVs; all 16 `\cite` keys resolve
  to `\bibitem`s (inline `thebibliography`); begin/end balanced; all `\ref` targets defined; the
  3 embedded figures exist.
- **All 16 references verified** against publisher records (July 2026) — no `TODO(verify)` left.
  Standard practice: re-confirm page numbers on final proofs.
- Author emails are placeholders except Sankari M; fill in real addresses.

## Compile

No LaTeX is installed in this environment, so the source is provided uncompiled. To build a PDF:

```bash
cd docs/paper
pdflatex main.tex && pdflatex main.tex     # twice, to resolve refs
```

Figures are pulled from `../../results/` via `\graphicspath`; they must exist first
(run the experiment scripts). Required figures:
`h1_collapse_recover.png`, `mechanism_forgetting.png` (and optionally `h2_deeplog.png`,
`h3_sequential.png`).

## For final submission

Switch `\documentclass[journal]{IEEEtran}` to the official **IEEE Access** template
(`ieeeaccess.cls`, downloadable from the IEEE Author Center). IEEEtran is used here only so the
source compiles with a standard TeX install. Then: finalize the author list, add citations
(reference paper, DeepLog, ECVC/ADSD, LogHub, Drain, FedAvg), and convert any remaining
`TODO` items once their runs land (see `docs/PAPER_DRAFT.md` "Remaining before submission").
