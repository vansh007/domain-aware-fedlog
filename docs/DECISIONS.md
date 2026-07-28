# Decisions log

> Every non-obvious choice, with rationale. This is where the paper's methodology
> justifications and threats-to-validity come from. Append-only.

## D1 — Scope to two datasets (HDFS, BGL)
**Decision:** Only HDFS and BGL. Thunderbird and Spirit are future work.
**Why:** The reference repo only supports HDFS/BGL. Adding domains means writing new
Drain parser configs and validating them — roughly a month of work with its own failure
modes. Two genuinely different domains are sufficient to test heterogeneity.
**Cost/threat:** Reviewers may ask for more domains. Pre-empt in limitations: two domains
already exhibit the failure; more domains would only widen the gap we predict.

## D2 — Keep our code separate from the GPL reference code
**Decision:** Call the reference scripts as a subprocess / import narrowly, rather than
copying their source into ours.
**Why:** Their code is GPL-3.0. Copying it into our files spreads GPL obligations. A
clean boundary keeps our wrappers and novel aggregation under our own chosen license and
makes the contribution boundary obvious to reviewers.
**Cost/threat:** Slightly more integration glue. Acceptable.

## D3 — Merged global vocabulary via ID offset
**Decision:** Build one global event-ID space across domains by offsetting: HDFS keeps
IDs 0..32, BGL takes 33..426. A `domain` tag travels with each sequence.
**Why:** This mirrors their step-3 "global template dictionary," just across datasets
instead of across clients of one dataset. It is the minimal honest extension of their
pipeline to a mixed federation. Keeping a domain tag lets the domain-aware aggregation
route correctly and lets us measure per-domain F1.
**Cost/threat:** Assumes no template collisions across datasets (safe — Drain templates
are dataset-specific strings). Documented in `src/vocabulary.py`.

## D4 — Framed as falsifiable hypotheses, not a committed method
**Decision:** Lead with H1/H2/H3 as predictions; the aggregation fix is secondary.
**Why:** If the claims survive, we still have a publishable negative result. This avoids
the failure mode of committing to a "novel method" before knowing whether the problem it
solves is real.
**Cost/threat:** None. This is the safer research posture.

## D5 — Domain-aware aggregation kept deliberately simple
**Decision:** The intended fix is per-domain parameters (e.g. per-domain [min,max] for
Length; per-domain sub-models or routing) rather than a novel architecture.
**Why:** Semester timeline. A simple, well-motivated fix that ships beats a clever one
that misses the deadline. The empirical result carries the paper.
**Status:** Design not finalised — this is the Week-4 research task. See
`src/aggregation.py` stub.

<!-- Add D6, D7, ... as you make choices during the project. -->
