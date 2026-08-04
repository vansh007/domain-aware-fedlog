# TILLNOW — progress log

> Append-only. Newest entry at the top. One entry per working session.
> This is the handoff between sessions (and between you and Claude Code).
> Read this + CLAUDE.md at the start of every session instead of re-deriving state.

Format for each entry:
```
## YYYY-MM-DD — <session focus>
- Did: ...
- Found: ...
- Next: ...
- Blocked on: ... (if anything)
```

---

## 2026-08-05 (cont.) — Closed the domain-unknown routing open problem
- Did: Implemented `predict_domain_unknown` in src/aggregation.py (was NotImplementedError stub)
  with TWO tag-free routing strategies + an `infer_domains` helper. NO MoE/adapter/ML router — the
  disjoint vocabulary structure already encodes domain identity; we read it back off the ids.
  1. vocab_block (Strategy 1): infer domain from event-id block; majority vote; id outside all
     blocks -> "unknown domain" -> flagged anomalous.
  2. conservative (Strategy 2): no inference; flag only if length outside EVERY domain's range.
  `scripts/run_domain_routing.py` evaluates oracle vs both, 5 seeds -> results/domain_routing.csv.
- Found: vocab_block == oracle EXACTLY on every test seq and seed (route_acc 1.000+/-0.000, 0
  unknown; HDFS 0.5614+/-0.0236, BGL 0.0332+/-0.0435 — identical to oracle). Conservative COLLAPSES
  on HDFS (0.0000+/-0.0001): HDFS anomalies are long enough to fit inside BGL's wide range, so
  "outside every range" never fires — the same containment that causes H1. Conservative == oracle
  only on BGL (the widest domain). So inference IS needed, and the vocab block supplies it perfectly,
  tag-free. Open case = OVERLAPPING vocabularies (shared template) -> future work.
- Wired into main.tex: new Table tab:routing; updated sec:da "Removing the oracle tag" + sec:h1
  "fix needs no oracle tag" to the two-strategy comparison (replaces the single-router paragraph,
  now cites domain_routing.csv). Structural validation PASS (9 tables, refs/cites/figs/envs balanced).
- Next: verify citations; ieeeaccess.cls; compile. 3rd dataset still the open user decision.
- Blocked on: Nothing.

---

## 2026-08-05 — Publishability push #2: EWC vs replay, directional forgetting, figures
- Did: Two NEW real experiments (3 seeds each) + analytical + figure overhaul, to reduce the
  two biggest reviewer risks.
  1. EWC mitigation (`scripts/mitigation_ewc.py`): memory-free anti-forgetting vs replay. EWC
     removes forgetting ONLY at lambda=1e8 (forget +0.014+/-0.005, HDFS 0.690, BGL 0.752); at
     lambda<=1e4 no effect — needs tuning across 6 orders of magnitude. Replay works tuning-free
     at 10%. Honest trade-off, both fixes valid. `results/mitigation_ewc.csv`. New sec:ewc + tab:ewc.
  2. Reverse-order H3 (`scripts/h3_reverse_order.py`): forgetting is DIRECTIONAL. Forward
     HDFS->BGL embedding forgets +0.665; reverse BGL->HDFS does NOT forget (-0.103+/-0.006, BGL
     improves); scalar safe both ways. Failure = onboarding BROAD-vocab (BGL 394) AFTER NARROW
     (HDFS 33). `results/h3_reverse_order.csv`. New "Forgetting is directional" para + tab:reverse.
  3. H1 generality: added an analytical min-max containment condition — the collapse is general
     over any domain pair where one domain's normal-length support covers another's anomaly-length
     support, not specific to HDFS+BGL. Directly answers the "only 2 domains" concern for H1.
  4. Figures: fixed all overlaps + illegible values on the 3 existing figures (baseline-label
     collision in H1; missing byte labels + swamped x3.7 in H3; clipped 0.037 + on-line annotation
     in mechanism); regenerated at 200dpi with white-boxed labels. Added 2 NEW figures:
     two_by_two_map.png (signature 2x2), replay_mitigation.png (buffer-size curve). `plot_results.py`
     gained kinds `map` and `replay`.
  5. Wired all into main.tex; updated abstract, contributions (iii), future work, map caption.
     Structural validation PASSED: refs/cites/bibitems/figures/envs all balanced (no local TeX to compile).
- Found: Both new results are clean and strengthen the paper (diagnose->explain->fix->compare, plus a
  directional refinement). No prior result reversed.
- Next: verify the 3 TODO(verify) citations; swap to ieeeaccess.cls; compile PDF. The one big
  remaining lever is a genuine 3rd dataset (needs a Drain parser = the CLAUDE.md-scoped "separate
  month"); flagged to the user as a decision, not silently done.
- Blocked on: Nothing.

---

## 2026-07-26 (cont.) — Publishability upgrades: deployable H1 router + replay fix
- Did: Honest novelty review of the paper, then added two REAL in-scope experiments to raise it
  from "diagnosis-only" to "diagnose -> explain -> FIX".
  1. Inferred domain router (`scripts/h1_inferred_router.py`): H1 fix routes by INFERRING domain
     from event-id vocab block instead of an oracle tag. 5 seeds: routing acc 1.0000+/-0.0000,
     inferred F1 == oracle F1 exactly (HDFS 0.5614+/-0.0236). Fix is deployable w/o any label.
     `results/h1_inferred_router.csv`.
  2. Replay mitigation (`scripts/mitigation_replay.py`): a small replay buffer fixes the
     embedding-DeepLog sequential forgetting. 3 seeds: no-replay forget +0.6647; >=10% buffer
     (~558 seqs) -> forget +0.0035+/-0.0035, HDFS restored 0.700, BGL still learned 0.722; 5%
     unreliable. Model stays fixed-size. `results/mitigation_replay.csv`.
- Wired both into main.tex: new sec:replay + Table tab:replay; sec:da + sec:h1 router paragraphs;
  abstract, contributions (ii)/(iii), discussion guidance (iii), future work all updated.
  Re-validated: refs/cites/figures/tables all consistent.
- Found: Both fixes work cleanly and honestly. Paper now has two self-contained contributions
  (a deployable aggregation fix + a demonstrated forgetting fix), not just a diagnosis.
- Next: writing only (author emails; ieeeaccess.cls; compile). Optional: EWC-vs-replay comparison.
- Blocked on: Nothing.

---

## 2026-07-26 — Centralized DeepLog capacity-ceiling run (pins the 0.93 gap)
- Did: Ran STANDALONE centralized DeepLog on FULL HDFS normal (558,223 seqs, no FedAvg, matched
  hyperparams, transductive eval). `scripts/deeplog_centralized_hdfs.py` — imports DeepLog/evaluate
  read-only, touches no federated script/result. Output `results/deeplog_centralized_hdfs.csv`.
  Reached round 29/100 (~3.5 h, machine throttled) and accepted as converged (loss flatlined).
- Found: Does NOT approach 0.93. Loss 0.349->0.204 (flat since ~r20). F1 PEAKS at 0.4333 @ r2 then
  DECLINES to 0.2548+/-0.0182 (r10-29); precision saturates ~0.97, recall stays <0.30. Confirms the
  gap is a scalar-input DeepLog + top-g capacity/method CEILING, not federation or 1%-subsampling.
  (Transductive ~0.25 is a different eval population than the held-out ~0.70 federated number — the
  DIRECTION is the point.) Wired into RESEARCH_LOG 5d + Threats-to-Validity in main.tex; removed the
  now-completed "centralized DeepLog run" future-work item.
- Next: writing only (author emails, ieeeaccess.cls, compile).
- Blocked on: Nothing.

---

## 2026-07-25 (cont.) — Full IEEE Access paper written
- Did: Wrote the complete paper. Web-researched related work (real citations), built
  `docs/paper/references.bib` (16 entries), and wrote the full `docs/paper/main.tex`:
  Abstract, Intro, Related Work (with genuine comparisons to Pustozerova2026, DeepLog,
  LogAnomaly, FedLAD 2025, the federated deep-log open-challenges paper, non-IID FL, FL open
  problems, EWC/continual+federated forgetting, Drain, Loghub, surveys), Problem Setting,
  Method, Setup, H1/H2/H3/Mechanism/Ensemble results, Discussion, Threats (incl. honest DeepLog
  gap), Conclusion. Authors: Vansh Mundhra, Pinak Debnath, Sankari M (VIT Chennai).
- Verified: every number matches the CSVs; all 15 \cite keys resolve; begin/end balanced; all
  \ref targets defined; 3 figures present. IEEEtran base (compiles anywhere); swap to
  ieeeaccess.cls for submission.
- Next: proof-check 3 TODO(verify) citations (ecvc2024, fedlog2024openchallenges,
  cfl2023quantifying); fill author emails; compile with ieeeaccess.cls.
- Blocked on: Nothing.

---

## 2026-07-25 (cont.) — DeepLog reproduction gap investigated (NOT hidden)
- Did: Reconciled the DeepLog 0.70-vs-reference-0.93 gap on request.
  - Exact config our runs used: 12 rounds x 1 local epoch, Adam lr=1e-3, batch=1024, no early
    stopping / no convergence check (fixed budget).
  - Reference config uses number_rounds=1, max_epoch=1 on 1% data — LESS training than ours.
  - Convergence sweep 1..100 rounds (`deeplog_convergence_le1_b1024.csv`): F1 plateaus by round
    5, best 0.7205 @100. Does NOT approach 0.93.
  - Top-g sweep (`deeplog_num_candidates_sweep.csv`): best F1=0.704 @g=9; recall hard-capped
    ~0.63 for all g. No threshold reaches 0.93.
- Found: Our federated DeepLog CANNOT reach 0.93; plateaus ~0.70. Likely reason: ~0.93 is the
  reference's CENTRALIZED/best-tuned number, not the federated-1%-data operating point we (and
  their federated config) implement; secondary recall ceiling from scalar-id input + 1% data.
  Documented honestly in RESEARCH_LOG 5c + Threats-to-Validity (PAPER_DRAFT + main.tex).
  H2/H3 already ran at rounds=12 (= plateau); forgetting is a relative effect, unaffected.
- Next: writing only (author list, citations, ieeeaccess.cls).
- Blocked on: Nothing.

---

## 2026-07-25 (cont.) — mechanism multi-seed + ensemble + LaTeX
- Did: Closed the remaining follow-ups.
  1. Multi-seed mechanism (3 seeds): H3 embedding forgetting +0.6647+/-0.0032 (HDFS F1 ->
     0.039+/-0.003); H2 embedding HDFS drop -0.0025+/-0.0018. Rock-solid.
     (`results/mechanism_multiseed.csv`, per-seed CSVs.)
  2. Ensemble lift: wrote fresh ECVC (`src/methods/ecvc.py`) + union runner
     (`scripts/run_ensemble.py`). Reproduced reference ensemble (HDFS single-domain F1=0.984).
     FINDING: domain-aware fix does NOT lift the union ensemble (0.984 either way) because ECVC
     (0.968, recall ~1.0) dominates the OR. So H1's fix matters for LIGHTWEIGHT-ONLY
     deployments, not the full ensemble — honest scoping, no overclaim. (`results/ensemble.csv`.)
  3. LaTeX: `docs/paper/main.tex` (IEEE format, all tables/figures wired) + `docs/paper/README.md`.
- Found: Nothing reversed under replication. The paper's story is complete and defensible.
- Next (all optional / writing): author list + citations; switch to ieeeaccess.cls + compile;
  domain-unknown routing (open Week-4 Q in aggregation.py); replay-based mitigation for
  embedding-DeepLog forgetting.
- Blocked on: Nothing. Experiment phase DONE.

---

## 2026-07-25 — H1 polish + H2 + H3 + paper draft
- Did: Closed out the remaining experiment plan end-to-end.
  1. H1 figure: `results/h1_collapse_recover.png` (collapse→recover bar chart, baseline line).
  2. Non-IID H1 (lognormal split, 5 seeds): H1 holds; domain-aware Length is partition-
     invariant (`results/h1_multiseed_lognormal.csv`, `..._lognormal.png`).
  3. Week-1 baselines logged: HDFS Length 0.5379 (ref ~0.57), BGL Known Events 0.9915
     (ref ~0.99) — gate PASSED (`results/{hdfs,bgl}_baseline.csv`).
  4. Installed CPU torch; wrote our own DeepLog + FedAvg (`src/methods/deeplog_fed.py`,
     `scripts/run_h2_deeplog.py`) — no flwr, no GPL copy.
  5. H2 (single seed): NEGATIVE — DeepLog does NOT degrade under mixing (HDFS drop +0.0002,
     BGL improves). `results/h2_deeplog.csv`, `.png`. Multi-seed (3) running.
  6. H3 (`scripts/run_h3_sequential.py`): set-union forgetting 0 but size ×3.7 (SUPPORTED);
     DeepLog forgetting 0, size fixed (forgetting NOT observed). `results/h3_sequential.csv`, `.png`.
  7. Paper draft: `docs/PAPER_DRAFT.md` with all real numbers wired; detailed lab notebook
     in `docs/RESEARCH_LOG.md`.
- Found (the key scientific twist): the deep-model negatives were caused by DeepLog's scalar
  event-id input (HDFS ids [0,32] vs BGL [33,426] = disjoint ranges). MECHANISM TEST PROVED IT:
  with a learned embedding (shared feature space), H3 sequential arrival CATASTROPHICALLY
  forgets HDFS (F1 0.7037 -> 0.0367, forgetting +0.667) vs 0.000 with scalar. BUT H2
  simultaneous mixing stays robust even with embeddings (HDFS drop -0.003). So the real
  distinction is simultaneous (safe) vs sequential (catastrophic) x disjoint vs shared repr —
  a clean 2x2 where only (sequential, shared) fails. Figure: results/mechanism_forgetting.png.
- Multi-seed H2 confirmed the negative: HDFS drop +0.0002+/-0.0001, BGL -0.1058+/-0.0023 (n=3).
- Next: (a) multi-seed the embedding H3 (1 seed so far; effect huge so unlikely to reverse);
  (b) optional ensemble lift (does domain-aware Length raise Events+Length+ECVC toward 0.95?);
  (c) prose polish + LaTeX conversion of docs/PAPER_DRAFT.md.
- Blocked on: Nothing.

---

## 2026-07-22 — H1 experiment: mixed HDFS+BGL federation (SUPPORTED)
- Did: Ran `configs/hdfs_bgl_mixed.yaml` on real data (IID, 3 HDFS + 2 BGL clients,
  merged vocab 427). Then multi-seeded it across seeds 0-4 via new `scripts/h1_multiseed.py`.
- Found — H1 SUPPORTED. Single seed (seed 0): `length_global` HDFS F1=0.0002 vs
  `length_domain_aware` HDFS F1=0.5379 — the domain-blind global min/max range collapses
  HDFS detection to ~zero; the per-domain (domain-aware) fix recovers it (~2700x).
  BGL identical under both methods (0.1183) — contamination is ASYMMETRIC: BGL is the
  domain widening the range, so it is unaffected; HDFS is the victim, so it is repaired.
  `known_events` (set-union) unaffected by mixing (HDFS 0.7885, BGL 0.9841) — the
  contrast case: only the range-based method breaks, which pinpoints the mechanism.
- Found — multi-seed (n=5, seeds 0-4): the effect is robust, not a single-seed fluke:
    length_global      HDFS  F1 mean 0.0000 +/- 0.0001   (collapse in every seed)
    length_domain_aware HDFS F1 mean 0.5614 +/- 0.0236   (recovery in every seed)
    length_global == length_domain_aware on BGL in ALL 5 seeds (asymmetry confirmed)
    known_events        HDFS  F1 mean 0.5636 +/- 0.1170  (high variance — see caveat)
    known_events        BGL   F1 mean 0.9866 +/- 0.0034
- Caveats (for the paper): (1) 0.54 is a RECOVERY of the Length *component*, not the
  ~0.95 full-ensemble number — the Week-4 question is whether fixing the component lifts
  the ensemble. (2) known_events HDFS is noisy across seeds (0.44-0.79); worth a note but
  not the H1 headline. (3) DeepLog (H2) not yet run — SKIPPED, no fake number written.
- Results: `results/hdfs_bgl_mixed.csv` (seed 0), `results/hdfs_bgl_mixed_seed{0..4}.csv`,
  `results/h1_multiseed.csv` (mean/std summary). Full narrative in `docs/RESEARCH_LOG.md`.
- Next: Wire DeepLog for H2 (`src/methods/deeplog_stub.py`) — needs torch+flwr installed
  in .venv first. Then re-run `hdfs_bgl_mixed.yaml` with deeplog enabled.
- Blocked on: torch/flwr not in .venv (reference needs torch==2.6.0, flwr==1.12.0).

---

## 2026-07-22 — Dataloader wired to real data
- Did: Wired `_load_parsed_domain(domain, data_root)` in `src/dataloader.py` to read the
  reference repo's CSVs at `{data_root}/{DOMAIN}/{normal,abnormal}.csv` (DOMAIN uppercase).
  Column 1 (block/node id) ignored; column 2 split into event strings; normal.csv=label 0,
  abnormal.csv=label 1. Return contract unchanged. `vocabulary.py`/`metrics.py` untouched.
- Found: Verified on real data. HDFS alone = 33 distinct events / 558223 normal / 16838
  abnormal (matches paper's ~33 / ~558k / ~16.8k). Merged HDFS+BGL vocab = 427 (33 + 394).
  Collision check passes: HDFS "5" -> global id 0, BGL "5" -> global id 43 (distinct).
- Notes:
  - Event-id collision fix: each event is represented as `f"{domain}:{int_id}"` so HDFS "5"
    and BGL "5" stay distinct in the merged vocabulary.
  - CSV field-limit fix: BGL's long sequences overflow csv's default per-field cap, so
    `_ensure_csv_field_limit()` lifts it to the platform max.
  - Convention: events are lowercase-prefixed (`hdfs:5`, `bgl:5`) matching the domain keys
    used across vocab/config, while the on-disk directories are uppercase (`HDFS`, `BGL`);
    `domain.upper()` bridges the two.
- Next: Run the mixed federation experiment (H1/H2) via `configs/hdfs_bgl_mixed.yaml`.
- Blocked on: Nothing.

---

## 0000-00-00 — Scaffold created
- Did: Set up project structure, configs, docs, working skeletons for vocabulary,
  dataloader, simple methods, and metrics. Wrote the four Claude context files.
- Found: N/A — nothing run yet.
- Next: WEEK 1. Clone the reference repo, install deps, download HDFS + BGL from LogHub,
  run `scripts/reproduce_baseline.py`. Do not proceed past this until the reference
  numbers reproduce.
- Blocked on: Nothing. First action is the Docker/venv build + one HDFS run.

<!--
  WEEK-1 GATE REMINDER: if you cannot reproduce the reference paper's HDFS/BGL numbers,
  the correct action is to STOP and report it to your supervisor. That is a finding.
  Do not silently patch numbers to make them "look right."
-->
