# PAPER HANDOFF — everything needed to write the paper

> **Read this first.** This is a self-contained brief for the LLM/author writing the final
> IEEE Access paper. Every number below is real and traces to a file in `results/`. **Do not
> invent, round-up, or "improve" any number.** If a value is not here or in `results/`, it does
> not exist — write `TODO(measure)`, do not fabricate.

---

## 0. Is the research over? What is left?

**The experimental work is DONE.** All hypotheses were tested, replicated across seeds, and a
mechanism experiment was run. **What remains is writing only:**
1. Related-work prose + citations (this doc gives you the reference list and the comparisons).
2. Polishing each section (structure + draft already exist in `PAPER_DRAFT.md` / `main.tex`).
3. Switching the LaTeX to the official IEEE Access class and compiling.
4. Author list / affiliations.

No new experiments are required to submit. Optional strengtheners are listed in §8.

---

## 1. What to hand the paper-writing LLM (in priority order)

**Tier 1 — must give (the paper is built from these):**
- `docs/paper/HANDOFF.md` ← this file (the master brief)
- `docs/paper/main.tex` ← the LaTeX draft with all tables/figures already wired
- `docs/PAPER_DRAFT.md` ← the Markdown draft (prose is a bit fuller than the .tex in places)
- `docs/RESEARCH_LOG.md` ← the lab notebook: every experiment, command, number, and problem
- `docs/REFERENCE_PAPER_NOTES.md` ← distilled facts about the paper we extend
- `docs/RULES_AND_GOALS.md` + `docs/DECISIONS.md` ← the "why" and the methodology justifications

**Tier 2 — the evidence (cite these; numbers must match):**
- All 5 figures: `results/h1_collapse_recover.png`, `results/h1_collapse_recover_lognormal.png`,
  `results/h2_deeplog.png`, `results/h3_sequential.png`, `results/mechanism_forgetting.png`
- The summary CSVs (see §3 for what each proves): `h1_multiseed.csv`,
  `h1_multiseed_lognormal.csv`, `h2_multiseed.csv`, `h3_sequential.csv`,
  `h3_sequential_embedding.csv`, `mechanism_multiseed.csv`, `ensemble.csv`,
  `hdfs_baseline.csv`, `bgl_baseline.csv`, `deeplog_convergence_le1_b1024.csv`,
  `deeplog_num_candidates_sweep.csv`

**Tier 3 — code (for the reproducibility appendix / methods precision):**
- `src/` (dataloader.py, vocabulary.py, aggregation.py, metrics.py,
  methods/{length,known_events,ecvc,deeplog_fed}.py)
- `scripts/` (per-hypothesis runners + plot_results.py)

---

## 2. The one-paragraph thesis (what the paper says)

Prior work [Pustozerova2026] reports that in *federated* log anomaly detection, lightweight
statistical detectors are (C1) approximately invariant to client data distribution and (C2)
bounded in size by a finite event vocabulary — but only tested this where every client holds a
slice of the **same** dataset (quantity skew). We test a **heterogeneous** federation (HDFS +
BGL clients over one merged 427-event vocabulary) and a **sequentially evolving** one. Findings:
(H1) domain-blind range aggregation collapses on the tighter domain and a simple **domain-aware**
aggregation restores it; (H2) simultaneous mixed FedAvg on the deep model does **not** degrade;
(H3) but **sequential** arrival causes **catastrophic forgetting** in the deep model — and a
controlled experiment shows this happens **only when the input representation makes the two
vocabularies share a feature space**, while set-union detectors never forget but grow unbounded.
The result is a precise 2×2 map of when federated log anomaly detection is safe vs. fails, plus a
working fix for the one aggregation failure.

---

## 3. ALL FINDINGS WITH EXACT NUMBERS (self-contained)

### 3.1 Reproduction gate (single-domain baselines) — `hdfs_baseline.csv`, `bgl_baseline.csv`
| Domain | Method | Our F1 | Reference | Verdict |
|---|---|---|---|---|
| HDFS | Length | 0.5379 | ~0.57 | matches |
| HDFS | Known Events | 0.7885 | (ensemble ~0.95) | plausible |
| BGL | Length | 0.0203 | weak | matches |
| BGL | Known Events | 0.9915 | ~0.99 | matches |

### 3.2 H1 — range aggregation collapses; domain-aware fixes it — `h1_multiseed.csv` (5 seeds, IID)
| Method | HDFS F1 | BGL F1 |
|---|---|---|
| Length, global (domain-blind) | 0.0000 ± 0.0001 | 0.0332 ± 0.0435 |
| Length, domain-aware (OURS) | **0.5614 ± 0.0236** | 0.0332 ± 0.0435 |
| Known Events (set-union) | 0.5636 ± 0.1170 | 0.9866 ± 0.0034 |
| (HDFS single-domain Length reference line) | 0.5379 | — |
- **Non-IID (log-normal) — `h1_multiseed_lognormal.csv`:** HDFS Length 0.0001 → 0.5614; story
  unchanged. Domain-aware Length is **partition-invariant** (identical IID vs non-IID).
- Interpretation: BGL identical under both aggregations = contamination is **asymmetric** (BGL
  widens the range, HDFS is the victim). Known Events unaffected = mechanism is specific to
  range-based aggregation. Figure: `h1_collapse_recover.png`.

### 3.3 H2 — simultaneous mixed FedAvg does NOT degrade DeepLog — `h2_multiseed.csv` (3 seeds)
| Domain | Single-domain | Mixed | Drop (single − mixed) |
|---|---|---|---|
| HDFS | 0.7039 | 0.7037 | +0.0002 ± 0.0001 |
| BGL | 0.6459 | 0.7517 | −0.1058 ± 0.0023 (mixed BETTER) |
- Negative result (strengthens C-generality for the deep model). Holds even with an embedding
  input (see §3.5). Figure: `h2_deeplog.png`.

### 3.4 H3 — sequential arrival — `h3_sequential.csv` (+ `_embedding.csv`)
| Method / input | HDFS F1 after HDFS | after BGL | Forgetting | Size after HDFS → BGL |
|---|---|---|---|---|
| Known Events | 0.7885 | 0.7885 | 0.0000 | 1,688 → 6,300 B (**×3.7**) |
| DeepLog (scalar id) | 0.7037 | 0.7037 | 0.0000 | 312,748 B (fixed) |
| DeepLog (embedding) | 0.7037 | **0.0367** | **+0.6670** | 355,436 B (fixed) |
- Set-union: never forgets but **grows unbounded** (bounded-size claim fails under evolution).
- DeepLog forgetting is **representation-dependent**. Figure: `h3_sequential.png`.

### 3.5 MECHANISM (the paper's key twist) — `mechanism_multiseed.csv` (3 seeds)
| Experiment | Metric | Value |
|---|---|---|
| H3 embedding | DeepLog HDFS forgetting | **+0.6647 ± 0.0032** (F1 → 0.039 ± 0.003) |
| H2 embedding | DeepLog HDFS drop | −0.0025 ± 0.0018 (still no degradation) |
| H2 embedding | DeepLog BGL drop | −0.1120 ± 0.0064 |
- **The 2×2:** {simultaneous mixing, sequential arrival} × {disjoint-range input, shared-repr
  input}. **Only (sequential, shared representation) fails** — and it fails hard (0.70→0.04).
  Scalar-id input keeps HDFS ids [0,32] and BGL ids [33,426] in disjoint ranges, masking the
  danger. Figure: `mechanism_forgetting.png` (the smoking-gun figure).

### 3.6 Ensemble lift — `ensemble.csv` (does fixing Length raise Events+Length+ECVC?)
| HDFS ensemble (Events ∨ Length ∨ ECVC) | F1 |
|---|---|
| Single-domain (reference ~0.95) | 0.9836 |
| Mixed, Length **global** (collapsed 0.0002) | 0.9837 |
| Mixed, Length **domain-aware** (recovered 0.5379) | 0.9836 |
- **Honest finding:** the fix does NOT change the ensemble, because it's a logical OR and ECVC
  alone is 0.968 (recall ~1.0) on HDFS. So H1's fix matters for **lightweight-only** deployments,
  not the full ensemble. Do NOT overclaim that mixing breaks their best method — it does not.

### 3.7 DeepLog reproduction gap — MUST be stated as a limitation (do not hide)
- Our federated HDFS DeepLog = **~0.70**, reference reports **~0.91–0.93**.
- Training sweep 1→100 rounds (`deeplog_convergence_le1_b1024.csv`): F1 0.087/0.704/0.704/0.698/
  0.671/**0.7205** — plateaus by round 5; more training does NOT reach 0.93.
- Threshold sweep (`deeplog_num_candidates_sweep.csv`): best 0.704 at g=9; **recall hard-capped
  ~0.63** for all g. No threshold reaches 0.93.
- **Likely reason:** ~0.93 is most plausibly the reference's **centralized/best-tuned** DeepLog,
  not the **federated-1%-data** operating point (their federated config trains only 1 round × 1
  epoch). Our number matches the federated regime. This is a *relative* comparison for H2/H3, so
  the forgetting conclusion is unaffected — but the paper must say we operate at ~0.70, not 0.93.

---

## 4. Related work to cite AND compare against

You are extending [Pustozerova2026]. Position the paper against these clusters. (Verify exact
venue/year/pages before final submission; keys below are suggestions.)

**A. The paper we extend (anchor citation).**
- **[Pustozerova2026]** Pustozerova, García Gómez, Landauer, Wurzenberger, Skopik, Weippl, Mayer,
  Ekelhart. "Lightweight Techniques for Federated Anomaly Detection in Log Data." *IEEE Trans.
  Reliability*, vol. 75, 2026. DOI 10.1109/TR.2026.3677674.
  → *Compare:* we use the same methods/datasets/pipeline; our novelty is the mixed + sequential
  federation they never test (their repo has no mixed config). Adopt their 1%-train, seed-avg
  discipline. Their own threats-to-validity (single-machine timing, no cross-study metric
  comparison) support ours.

**B. Deep log anomaly detection (H2/H3 subjects).**
- **[Du2017]** Du, Li, Zheng, Srikumar. "DeepLog: Anomaly Detection and Diagnosis from System
  Logs through Deep Learning." *ACM CCS*, 2017. → the model we federate; note the top-g rule.
- **[Meng2019]** Meng et al. "LogAnomaly: Unsupervised Detection of Sequential and Quantitative
  Anomalies in Unstructured Logs." *IJCAI*, 2019. → the other deep baseline in [Pustozerova2026].

**C. Log parsing & datasets (setup).**
- **[He2017Drain]** He, Zhu, Zheng, Lyu. "Drain: An Online Log Parsing Approach with Fixed Depth
  Tree." *IEEE ICWS*, 2017. → the parser producing event templates.
- **[He2016Loglizer]** He, Zhu, He, Lyu. "Experience Report: System Log Analysis for Anomaly
  Detection." *IEEE ISSRE*, 2016. → HDFS/BGL benchmarking; motivates method choices.
- **[Zhu2023Loghub]** Zhu, He, et al. "Loghub: A Large Collection of System Log Datasets."
  *IEEE ISSRE*, 2023. → source of HDFS and BGL.

**D. Federated learning (the FL half).**
- **[McMahan2017]** McMahan, Moore, Ramage, Hampson, Arcas. "Communication-Efficient Learning of
  Deep Networks from Decentralized Data." *AISTATS*, 2017. → FedAvg (the H2/H3 aggregation).
- **[Kairouz2021]** Kairouz et al. "Advances and Open Problems in Federated Learning." *Found. &
  Trends in ML*, 2021. → heterogeneity taxonomy; position DOMAIN skew as under-studied.
- **[Zhao2018]** Zhao, Li, Lai, Suda, Civin, Chandra. "Federated Learning with Non-IID Data."
  arXiv:1806.00582, 2018. → shows weight-averaging degrades under non-IID; contrast: our H2 does
  NOT degrade under *domain* skew for scalar-input DeepLog (and we explain why).

**E. Catastrophic forgetting / continual learning (H3 framing).**
- **[McCloskey1989]** McCloskey, Cohen. "Catastrophic Interference in Connectionist Networks."
  *Psychology of Learning and Motivation*, 1989. → the phenomenon.
- **[Kirkpatrick2017]** Kirkpatrick et al. "Overcoming Catastrophic Forgetting in Neural
  Networks (EWC)." *PNAS*, 2017. → mitigation; cite as future work for our embedding-DeepLog.
- **[DeLange2022]** De Lange et al. "A Continual Learning Survey: Defying Forgetting in
  Classification Tasks." *IEEE TPAMI*, 2022. → frames sequential domain arrival as continual FL.

**F. ECVC / count-vector method (ensemble member).**
- **[ECVC/ADSD]** The count-vector anomaly method referenced by [Pustozerova2026] as ECVC,
  DOI 10.1145/3660768 (ACM, 2024). → cite for the ECVC member we reimplement.

**G. (Optional) Federated anomaly / intrusion detection** — 1–2 recent surveys on federated
anomaly detection to show the sub-field is active and domain-heterogeneity is open.

**Novelty sentence to defend against reviewers:** "To our knowledge, this is the first study of
*domain* heterogeneity (not quantity skew) and *sequential domain arrival* in federated log
anomaly detection, and the first to isolate input representation as the variable that decides
whether a federated deep log model forgets."

---

## 5. Section-by-section plan (map findings → sections)

1. **Abstract** — done in `main.tex`; keep the 2×2 framing.
2. **Introduction** — motivation (privacy-preserving log analytics), the C1/C2 claims, the gap
   (no mixed/sequential config), contributions (merged vocab; H1 failure+fix; H2 negative; H3
   representation-dependent forgetting + set-union growth). Cite A, D.
3. **Background & Related Work** — B (deep log AD), C (parsing/data), D (FL + non-IID taxonomy),
   E (continual learning). End by stating the gap and our position.
4. **Method** — 3.1 merged cross-domain vocabulary + collision handling (`domain:id`); 3.2
   federation construction (1% train, IID/log-normal); 3.3 detectors (Length, Known Events, ECVC,
   DeepLog with scalar vs embedding input); 3.4 domain-aware aggregation (the fix) + the open
   domain-unknown routing problem. Cite [Du2017] for DeepLog, [McMahan2017] for FedAvg.
5. **Experimental Setup** — datasets (§3 numbers), 3 HDFS + 2 BGL clients, 5 seeds, CPU, metrics.
   Include the reproduction-gate table (§3.1).
6. **Results** — one subsection per H1/H2/H3 + mechanism + ensemble, using §3 tables and the 5
   figures. Lead with H1 (the fix), then the H2/H3/mechanism 2×2, then the honest ensemble scoping.
7. **Threats to Validity** — the DeepLog ~0.70-not-0.93 limitation (§3.7, VERBATIM honesty),
   two-domains-only, representation-dependence, Known-Events seed variance, eval convention,
   transductive ECVC threshold.
8. **Conclusion & Future Work** — the 2×2 map; the fix; future: embedding-DeepLog with EWC/replay
   ([Kirkpatrick2017]); domain-unknown routing; more domains (Thunderbird/Spirit).
9. **Reproducibility appendix** — point to `results/` + `scripts/`; every number traces to a CSV.

---

## 6. Figures & tables manifest (ready to \includegraphics)

| Asset | File | Shows |
|---|---|---|
| Fig H1 | `results/h1_collapse_recover.png` | Length collapse + domain-aware recovery + baseline line |
| Fig H1 (non-IID) | `results/h1_collapse_recover_lognormal.png` | same under quantity skew |
| Fig H2 | `results/h2_deeplog.png` | DeepLog single vs mixed per-domain |
| Fig H3 | `results/h3_sequential.png` | size growth (set-union) + forgetting panels |
| Fig Mechanism | `results/mechanism_forgetting.png` | scalar vs embedding forgetting (smoking gun) |
| Table 1 | repro gate | §3.1 |
| Table 2 | H1 5-seed | §3.2 |
| Table 3 | H2 3-seed | §3.3 |
| Table 4 | H3 (incl. embedding row) | §3.4 |
| Table 5 | ensemble | §3.6 |
(All five tables already typeset in `main.tex`.)

---

## 7. HARD RULES for the writing LLM (non-negotiable)

1. **Never fabricate or round-up a number.** Use §3 / the CSVs verbatim. If it's not measured,
   write `TODO(measure)` — do not invent.
2. **State the DeepLog gap honestly** (§3.7). Do NOT imply we reproduced 0.93. Our DeepLog is the
   ~0.70 federated-1%-data operating point.
3. **Do NOT overclaim the ensemble result.** The domain-aware fix does not lift the full ensemble
   (ECVC dominates). Scope H1's impact to lightweight-only deployments.
4. **H2 is a NEGATIVE result** — frame it as strengthening the reference's generality, not as a
   failure of our work.
5. **The forgetting result is a RELATIVE effect** (0.70→0.04 vs 0.70→0.70), robust to the absolute
   DeepLog level.
6. Keep the honest asymmetry framing for H1 (BGL is the cause, HDFS the victim; damage and repair
   are one-sided).

---

## 8. Optional strengtheners (only if time; not required to submit)

- **EWC/replay mitigation** for embedding-DeepLog to complete the H3 continual-learning story
  ([Kirkpatrick2017]). Would turn "it forgets" into "it forgets, and here's a fix."
- **Domain-unknown routing** for domain-aware Length (the open Week-4 question in
  `src/aggregation.py::predict_domain_unknown`) — infer domain from event-id block at inference.
- **A third domain** (Thunderbird/Spirit) to widen the H1 gap — but needs a new Drain parser
  (out of current scope, `DECISIONS.md` D1).
- **Centralized DeepLog** run to confirm it reaches ~0.93 and pin the gap's cause definitively.

---

## 9. TL;DR for the human (you)

Give the other LLM the **six Tier-1 docs + five figures + the summary CSVs in §3**. Tell it to
follow §5 (section plan), use §3 numbers verbatim, cite §4, and obey §7 (hard rules). The science
is done and defensible; only the writing remains.
