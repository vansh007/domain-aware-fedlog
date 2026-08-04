# CLAUDE.md

> This file is read automatically by Claude Code at the start of every session.
> Keep it short. It exists to save tokens: instead of re-explaining the project
> each session, point Claude here. Update the "Current state" line whenever things move.

## What this project is (one paragraph)

We are testing whether two published properties of federated log anomaly detection
survive a **heterogeneous, evolving** federation. The reference paper (Pustozerova
et al., IEEE Transactions on Reliability, Vol. 75, 2026) claims (1) lightweight
statistical methods are *invariant to client data distribution*, and (2) their model
size is *bounded by a finite event vocabulary*. Both claims were only ever tested
where every client holds the **same** log dataset. We test what happens when HDFS
clients and BGL clients share **one** federation, and when new log domains arrive
**sequentially**. Target venue: IEEE Access.

## The one-sentence gap

Their public repo has `hdfs_iid.yaml`, `hdfs_no_iid.yaml`, `bgl_iid.yaml`,
`bgl_no_iid.yaml` — but **no `hdfs_bgl_mixed.yaml`**. That missing config is the paper.

## Current state

<!-- UPDATE THIS LINE EVERY SESSION -->
STATE: EXPERIMENT PHASE COMPLETE. H1 SUPPORTED (5 seeds; HDFS Length 0.000->0.561 via domain-aware fix, holds non-IID). H2 NEGATIVE (3 seeds; simultaneous FedAvg robust even w/ embeddings). H3: set-union grows x3.7 no-forget; DeepLog forgetting PROVEN representation-dependent (3-seed: embedding -> catastrophic +0.6647+/-0.0032, HDFS 0.70->0.04; scalar -> 0.00). Ensemble: fix does NOT lift union ensemble (0.984, ECVC dominates) -> scopes H1 to lightweight-only. Clean 2x2: only (sequential, shared-repr) fails. TWO FIXES ADDED 2026-07-26 (diagnose->explain->FIX): (a) inferred domain router — H1 fix routes by vocab-block id, no oracle tag, routing acc 1.000, inferred F1==oracle (results/h1_inferred_router.csv); (b) replay mitigation — >=10% buffer (~558 seqs) cuts embedding-DeepLog forgetting +0.665->+0.004 while BGL still learned, model fixed-size (results/mitigation_replay.csv). Paper: docs/PAPER_DRAFT.md + docs/paper/main.tex (LaTeX). Lab notebook: docs/RESEARCH_LOG.md. TWO MORE EXPERIMENTS ADDED 2026-08-05 (publishability push): (c) EWC vs replay — memory-free EWC removes forgetting ONLY at lambda=1e8 (forget +0.014, HDFS 0.690, BGL 0.752), needs 6-orders tuning vs replay's tuning-free 10%; honest trade-off (results/mitigation_ewc.csv); (d) reverse-order H3 (BGL->HDFS) — forgetting is DIRECTIONAL/asymmetric: embedding forgets HDFS forward (+0.665) but does NOT forget BGL reverse (-0.103, BGL improves); failure = broad-vocab-after-narrow only (results/h3_reverse_order.csv). Added H1 analytical generality argument (min-max containment condition, general over domains). New figures: two_by_two_map.png, replay_mitigation.png (+ all 3 old figures regenerated, overlaps fixed, 200dpi). Paper structurally validated (refs/cites/figs/envs all balanced; NOT compiled — no local TeX). DOMAIN-UNKNOWN ROUTING RESOLVED 2026-08-05: predict_domain_unknown in src/aggregation.py implemented with TWO tag-free strategies (no MoE/adapter/ML). (1) vocab_block — infer domain from event-id block, majority vote, id outside all blocks -> anomalous; (2) conservative — flag only if outside every domain's range. run_domain_routing.py (5 seeds): vocab_block == oracle EXACTLY on all 638,353 test seqs/seed (route_acc 1.000, 0 unknown, HDFS 0.5614, BGL 0.0332); conservative COLLAPSES on HDFS (0.000, anomalies fit inside BGL's wide range). results/domain_routing.csv. New Table tab:routing + updated sec:da/sec:h1 prose. Open case = OVERLAPPING vocabularies (future work). Remaining = verify citations, ieeeaccess.cls, compile. NOTE: our federated DeepLog operates at ~0.70 (plateau; best 0.7205 over 1-100 rounds), NOT the reference headline ~0.93 — that gap is real and documented (RESEARCH_LOG 5c/5d + Threats-to-Validity); ~0.93 is likely their centralized number. CONFIRMED 2026-07-26: standalone centralized DeepLog on FULL HDFS (no FedAvg, 100 passes) also does NOT reach 0.93 — loss flatlines 0.204, F1 peaks 0.43@r2 then settles ~0.25, recall <0.30 — so the gap is a scalar-input capacity/method ceiling, not a federation/subsampling artifact (results/deeplog_centralized_hdfs.csv). H2/H3 forgetting is a relative effect, unaffected.

## The three hypotheses (what we are actually testing)

- **H1**: In a mixed HDFS+BGL federation, range-based aggregation (Length) degrades
  sharply, dragging down ensembles that contain it (their best HDFS method,
  Events+Length+ECVC, has Length inside it).
- **H2**: In a mixed federation, FedAvg on deep models (DeepLog) degrades because
  client event vocabularies are largely disjoint (HDFS=33 types, BGL=394 types).
- **H3**: Under sequential arrival (HDFS -> BGL), lightweight models grow linearly
  while deep models forget earlier domains. Neither is deployable as-is.

If H1 and H2 come back NEGATIVE (everything works cross-domain), that is still a
publishable result strengthening their generality claim — pivot to H3 alone.
Do NOT force a positive result. Report what the experiments say.

## Hard rules (do not violate)

1. **Never fabricate experimental numbers.** Every metric in the paper must come
   from an actual run in `results/`. If asked to "fill in results," refuse and run
   the experiment instead, or leave a `TODO(measure)` marker.
2. **Reproduce before extending.** Week 1 gate: match the reference paper's HDFS and
   BGL numbers within reason before touching the mixed-domain code. If we cannot
   reproduce, that is a finding to report, not a bug to paper over.
3. **Two datasets only (HDFS, BGL).** Thunderbird/Spirit need new parsers = a separate
   month. They are future work, not scope.
4. **CPU-first.** The DeepLog baseline is tiny (2 layers x 64 units). Everything must
   run on a laptop CPU. Do not add code that assumes a GPU.
5. **The real work is `src/dataloader.py`.** The reference dataloader assumes one
   dataset per run. The novel contribution starts with a merged global vocabulary
   (HDFS event IDs 0-32, BGL event IDs 33-426). See `docs/RULES_AND_GOALS.md`.

## Where things live

- `configs/`        — YAML experiment configs (the mixed one is the point)
- `src/`            — dataloader, methods, aggregation, metrics
- `scripts/`        — run scripts, plotting, result gathering
- `experiments/`    — one markdown log per experiment run (append-only)
- `results/`        — CSVs and plots produced by runs (git-ignored, real data only)
- `docs/`           — the "why" files: goals, decisions, reference-paper notes
- `.claude/`        — extra context modules loaded on demand

## Key references (do not re-derive these each session)

- Reference paper repo: https://github.com/ait-aecid/comparison-fed-centr-efficient-ad
  (GPL-3.0, Python 3.12.3, uses Flower). Known issues: README mis-names
  `ml_centralize.py` (actual: `ml_centralised.py`); simulation script exits with a
  Flower exception after saving results (documented upstream, not our bug).
- Datasets: LogHub (https://github.com/logpai/loghub) — HDFS and BGL.
- Reference paper key numbers we must reproduce first:
  - HDFS: 33 event types; ~558k normal + ~16.8k abnormal sequences; avg length 29±6
  - BGL:  394 event types; ~37.8k normal + ~31.4k abnormal sequences; avg length 69±747
  - Their best HDFS method (Events+Length+ECVC) F1 ~0.95; DeepLog F1 ~0.91-0.93

## Token-saving conventions for future sessions

- Start a session by reading THIS file + `docs/TILLNOW.md`. Do not re-read the whole
  reference paper unless a specific claim is in question — the relevant facts are
  already summarised in `docs/REFERENCE_PAPER_NOTES.md`.
- When you finish work, append a dated entry to `docs/TILLNOW.md` and update the
  `STATE:` line above. That is the handoff.
- Long code belongs in files, not in chat. Point to the file path.
