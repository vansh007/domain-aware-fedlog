# RESEARCH_LOG — the paper's evidence trail

> Purpose: a precise, append-only record of **every experimental step, the result it
> produced, the problems hit, and how they were solved**. This is the raw material for the
> Methods, Results, and Threats-to-Validity sections of the paper. `TILLNOW.md` is the
> terse session handoff; **this file is the detailed lab notebook.**
>
> Rules that mirror CLAUDE.md: never fabricate a number; every metric here traces to a file
> in `results/`; if something is not yet measured it is marked `TODO(measure)`, not guessed.
>
> Newest section at the bottom (chronological — a lab notebook reads forward).

---

## 0. Setup & environment (facts the paper's reproducibility section needs)

- **Reference paper**: Pustozerova et al., *IEEE Transactions on Reliability*, Vol. 75, 2026.
- **Reference repo**: `reference_repo/` (github.com/ait-aecid/comparison-fed-centr-efficient-ad,
  GPL-3.0). DeepLog model itself is MIT (logdeep, d0ng1ee) — see
  `reference_repo/models/ml_models.py` header.
- **Datasets** (LogHub HDFS + BGL), living under `reference_repo/datasets/{HDFS,BGL}/`,
  symlinked to `data/{HDFS,BGL}` so configs can use `data_root: data`.
  - Format per domain: `normal.csv` (label 0) + `abnormal.csv` (label 1); header
    `ID,Event_seq`; col 1 = block/node id (ignored); col 2 = space-separated int event ids.
  - Row counts (verified `wc -l`, minus header):
    HDFS 558223 normal / 16838 abnormal; BGL 37823 normal / 31429 abnormal.
    These match the paper's ~558k/~16.8k and ~37.8k/~31.4k.
- **Compute**: laptop CPU only (CLAUDE.md hard rule 4).
- **Python env**: `.venv` has `pyyaml`; does **NOT** yet have `torch`/`flwr` (needed for
  DeepLog/H2). Reference pins `torch==2.6.0`, `flwr==1.12.0`.

---

## 1. Dataloader wired to real CSVs  (2026-07-22)

**Goal.** Make `src/dataloader._load_parsed_domain(domain, data_root)` read the real
reference CSVs and feed the merged-vocabulary pipeline. This is CLAUDE.md's "real work".

**What was done.** Read `{data_root}/{DOMAIN}/{normal,abnormal}.csv` (DOMAIN uppercased),
discard header, ignore col 1, split col 2 into per-event strings. Return
`(templates, examples)` unchanged per the contract.

**Problem 1 — event-ID collision.** HDFS and BGL both reuse small integers ("5", "6", "9",
...) for *different* templates. A naive int vocabulary would merge HDFS event 5 and BGL
event 5 into one global id — silently corrupting both domains.
**Solution.** Represent every event as `f"{domain}:{int_id}"` (`hdfs:5` vs `bgl:5`), so the
merged `GlobalVocabulary` keeps them distinct. Verified: `hdfs:5 -> 0`, `bgl:5 -> 43`.

**Problem 2 — CSV field-size overflow.** BGL sequences are very long (length std ~747), and
a single `Event_seq` field can exceed Python's default csv per-field cap (128 KB), raising
`_csv.Error: field larger than field limit`.
**Solution.** `_ensure_csv_field_limit()` lifts the cap to the platform max (backing off from
`sys.maxsize` on OverflowError). No rows dropped.

**Convention decided.** Events are lowercase-prefixed (`hdfs:`, `bgl:`) to match the domain
keys used across the vocab/config/aggregation layers; on-disk directories are uppercase
(`HDFS`, `BGL`). `domain.upper()` bridges the two.

**Verification (real data).**
- HDFS alone: **33** distinct events, **558223** normal, **16838** abnormal. Matches paper.
- Merged HDFS+BGL vocab: **427** = 33 (ids [0,33)) + 394 (ids [33,427)). Matches paper.
- Encoding sanity: `hdfs:5 hdfs:5 hdfs:5 hdfs:22 hdfs:11 hdfs:9 ...` -> `[0,0,0,1,2,3,...]`.

**Artifacts.** `src/dataloader.py`. No `results/` file (this step produces the loader, not a
metric). `vocabulary.py`/`metrics.py` untouched; return contract unchanged.

---

## 2. H1 — mixed HDFS+BGL federation, single seed  (2026-07-22)

**Hypothesis (H1).** In a mixed HDFS+BGL federation, range-based (Length) aggregation
degrades sharply on HDFS, because BGL's enormous length spread widens the shared global
[min,max] until almost no HDFS sequence looks anomalous.

**Setup.** `configs/hdfs_bgl_mixed.yaml`: IID split, 3 HDFS + 2 BGL clients, merged vocab
427, seed 0. Methods: `length` (domain-blind), `known_events` (set-union), plus OUR
`length_domain_aware` (per-domain range, routed by domain tag). DeepLog present in config
but SKIPPED (not wired). 1% of normal per domain used for training (reference convention).

**Command.** `python scripts/run_experiment.py --config configs/hdfs_bgl_mixed.yaml`

**Result (seed 0), `results/hdfs_bgl_mixed.csv`:**

| method                | domain | precision | recall | F1     |
|-----------------------|--------|-----------|--------|--------|
| length_global         | hdfs   | 0.087     | 0.0001 | 0.0002 |
| length_global         | bgl    | 0.9033    | 0.0633 | 0.1183 |
| known_events          | hdfs   | 0.9744    | 0.6622 | 0.7885 |
| known_events          | bgl    | 0.9687    | 1.0000 | 0.9841 |
| length_domain_aware   | hdfs   | 0.9930    | 0.3689 | 0.5379 |
| length_domain_aware   | bgl    | 0.9033    | 0.0633 | 0.1183 |

**Reading.**
- **H1 supported.** Domain-blind Length on HDFS: F1 **0.0002** (recall 0.0001 — it flags
  almost nothing). Our domain-aware fix: F1 **0.5379**. ~2700x recovery from the exact
  intervention the paper proposes.
- **Asymmetry (the clean mechanistic story).** BGL is **identical** under both methods
  (0.1183): BGL is the domain *widening* the range, not its victim, so domain-awareness
  neither helps nor hurts it. The damage is one-sided (HDFS) and so is the repair.
- **Contrast case.** `known_events` (set-union) is unaffected by mixing (HDFS 0.79, BGL
  0.98) — not every lightweight method breaks, only the range-based one. This specificity
  strengthens the claim: we pinpoint a mechanism rather than assert general collapse.

**Honest framing for the paper.** 0.54 is a **recovery of the Length component**, not the
~0.95 the full Events+Length+ECVC ensemble reaches. The mechanism is demonstrated; whether
fixing the component lifts the whole ensemble is Week-4 work.

---

## 3. H1 — multi-seed robustness (n=5)  (2026-07-22)

**Why.** One seed is one data point. "The effect is large so I didn't check" is not a
sentence for a paper. The reference authors averaged 3 seeds; we match the discipline with 5.

**What was built.** Added a `--seed` (+ `--suffix`) override to
`scripts/run_experiment.py` so a run can be reseeded without editing the config, and
`scripts/h1_multiseed.py`, which drives seeds 0-4 **through the exact same `run()` code
path** (no divergent implementation — important for validity) and aggregates mean/std.

**Command.** `python scripts/h1_multiseed.py`  (seeds 0 1 2 3 4)

**Result, `results/h1_multiseed.csv` (F1 mean +/- population std, n=5):**

| method                | domain | mean F1 | std    | per-seed values                     |
|-----------------------|--------|---------|--------|-------------------------------------|
| length_global         | hdfs   | 0.0000  | 0.0001 | 0.0002 0.0000 0.0000 0.0000 0.0000  |
| length_domain_aware   | hdfs   | 0.5614  | 0.0236 | 0.5379 0.5375 0.5886 0.5905 0.5524  |
| length_global         | bgl    | 0.0332  | 0.0435 | 0.1183 0.0085 0.0289 0.0015 0.0089  |
| length_domain_aware   | bgl    | 0.0332  | 0.0435 | 0.1183 0.0085 0.0289 0.0015 0.0089  |
| known_events          | hdfs   | 0.5636  | 0.1170 | 0.7885 0.5289 0.5284 0.5274 0.4447  |
| known_events          | bgl    | 0.9866  | 0.0034 | 0.9841 0.9912 0.9903 0.9834 0.9838  |

**Reading.**
- **H1 is robust, not a fluke.** HDFS Length: **0.0000 +/- 0.0001** (global) ->
  **0.5614 +/- 0.0236** (domain-aware). The collapse happens in all 5 seeds; the recovery
  happens in all 5 seeds; the recovery's std is tight (~0.024).
- **Asymmetry confirmed across every seed.** `length_global` and `length_domain_aware` are
  **identical on BGL in all 5 seeds** — the mechanism (BGL widens, HDFS suffers) is not
  seed-dependent. Note BGL's own Length F1 is low and variable (0.03 +/- 0.04) because
  Length was always weak on BGL; that is a property of BGL, not of mixing.
- **New caveat surfaced by multi-seed.** `known_events` on HDFS is noisy: **0.56 +/- 0.12**
  (0.44-0.79). The single-seed 0.79 was the high end. Worth a sentence in the paper; it does
  not affect the H1 headline (which is about Length), but it is exactly the kind of variance
  a single seed hides. This is *why* we multi-seed.

**Status.** H1 is solid enough to write up. Remaining H1 polish (optional): non-IID
(`split: lognormal`) to show the collapse isn't an artifact of even partitioning.

---

## 3b. Week-1 gate — single-domain baselines logged  (2026-07-25)

**Why now.** The reproduce-first rule (CLAUDE.md #2) requires single-domain numbers to
match the reference before mixed-domain conclusions stand. HDFS was mentioned earlier but
not in `results/`; BGL was never logged. Both are now recorded.

**Commands.** `python scripts/reproduce_baseline.py --dataset {hdfs,bgl}`

**Result** (`results/hdfs_baseline.csv`, `results/bgl_baseline.csv`, seed 0, 3 clients):

| domain | method        | F1     | reference expectation                 | verdict |
|--------|---------------|--------|---------------------------------------|---------|
| HDFS   | length_global | 0.5379 | "Length alone ~0.57"                  | matches |
| HDFS   | known_events  | 0.7885 | (their strong methods ~0.95 use ECVC) | plausible |
| BGL    | length_global | 0.0203 | "Length weak on BGL"                  | matches |
| BGL    | known_events  | 0.9915 | "Events ~0.99"                        | matches |

**Reading.** Lightweight-method gate **PASSED**: the right methods are strong/weak on the
right datasets, and the magnitudes line up (HDFS Length ~0.54 vs ref ~0.57; BGL Known
Events ~0.99 vs ref ~0.99). DeepLog still pending (H2). Note our `known_events` is not
their full ECVC ensemble, so we don't expect the 0.95 HDFS ensemble number here — that is
an ensemble result, out of scope for the single-method gate.

**Bonus insight (feeds the H1 narrative).** HDFS single-domain Length = **0.5379**. In the
mixed federation it collapses to **0.0000**, and domain-aware aggregation recovers it to
**0.5614** — i.e. the fix restores HDFS Length to *its own standalone number*. The recovery
target is the baseline, and we hit it. This is now the dashed reference line in the H1 plot.

---

## 3c. H1 figure + non-IID robustness  (2026-07-25)

**Figure.** Added `--kind h1` to `scripts/plot_results.py`; it reads the multi-seed summary
and draws the collapse-and-recover bar chart with mean±std error bars, overlaying the HDFS
single-domain Length baseline as a dashed line.
Output: `results/h1_collapse_recover.png` (IID), `results/h1_collapse_recover_lognormal.png`.

**Non-IID run.** To rule out "the collapse is an artifact of even (IID) partitioning", added
a `--split` override (`scripts/run_experiment.py` and `scripts/h1_multiseed.py`) and reran
the mixed federation with `split: lognormal` (quantity skew, sigma=0.25) across seeds 0-4.

**Command.** `python scripts/h1_multiseed.py --split lognormal`

**Result** (`results/h1_multiseed_lognormal.csv`, n=5):

| method                | domain | mean F1 | std    |
|-----------------------|--------|---------|--------|
| length_global         | hdfs   | 0.0001  | 0.0002 |
| length_domain_aware   | hdfs   | 0.5614  | 0.0236 |
| length_global         | bgl    | 0.0447  | 0.0510 |
| length_domain_aware   | bgl    | 0.0447  | 0.0510 |
| known_events          | hdfs   | 0.5636  | 0.1170 |
| known_events          | bgl    | 0.9910  | 0.0013 |

**Reading.** H1 holds under non-IID: HDFS Length still collapses (0.0001) and still recovers
(0.5614). Two extra observations worth a sentence in the paper:
- **Domain-aware Length is partition-invariant.** Its HDFS numbers are *identical* IID vs
  lognormal (0.5614±0.0236 both). Reason: the per-domain aggregated [min,max] is taken over
  the union of a domain's client data, which does not depend on how that data is split across
  clients. So the fix is robust to quantity skew *by construction* — a small but clean point.
- **BGL Length varies a bit more under skew** (0.045 vs 0.033) but stays low; unchanged story.

---

## 4. H2 — DeepLog under FedAvg over disjoint vocabularies  (2026-07-22, IN PROGRESS)

**Hypothesis (H2).** FedAvg on DeepLog degrades in a mixed federation because HDFS
(ids 0-32) and BGL (ids 33-426) event spaces are largely disjoint, so weight-averaging
blends models that learned different vocabularies.

**Blocker identified.** `.venv` lacks `torch` and `flwr` (reference pins torch==2.6.0,
flwr==1.12.0). DeepLog is a 2-layer x 64-unit LSTM (`reference_repo/models/ml_models.py`,
class `deeplog`) — tiny, CPU-friendly, but PyTorch is required. `run_experiment.py`
currently prints `deeplog: SKIPPED` and writes **no** fake number (correct behaviour).

**Plan (next concrete step).**
1. Install CPU-only torch + flwr into `.venv`.
2. Reuse the reference `deeplog` nn.Module (MIT-licensed) rather than reimplementing.
3. Windows: HDFS window_size=10, BGL window_size=2 (reference paper values); top-g
   candidates for the anomaly rule (HDFS ~9, BGL ~60).
4. Minimal faithful FedAvg over our federation's per-client encoded sequences (CPU), then
   per-domain evaluation via `src.metrics.prf`. Evaluation over the full ~570k HDFS test
   set is the expensive part on CPU; first pass may cap/subsample the test pool and record
   that clearly as a `TODO(measure: full test set)`.

**What was built (2026-07-25).** Installed CPU `torch` (2.13.0) into `.venv` — did NOT need
`flwr`: rather than drive the reference's Flower simulation (GPL, plus its documented exit
bug), we reuse only the DeepLog *mechanism* and write our own FedAvg. New files:
`src/methods/deeplog_fed.py` (fresh 2x64 LSTM, sliding-window training, top-g eval, weighted
FedAvg) and `scripts/run_h2_deeplog.py`. Hyper-params match the reference (window=10,
candidates=9, hidden=64, layers=2). Eval is made tractable by deduplicating identical test
sequences (HDFS 569k -> 26.7k unique; BGL 68k -> 27.5k unique) and weighting by count.

**Adaptation.** Reference subtracts 1 (their ids are 1-indexed); our merged vocab is already
0-indexed, so we feed ids directly with num_classes = vocab_size (33 single-HDFS, 394
single-BGL, 427 mixed). Short sequences (< window+1) are treated as normal; the single-vs-
mixed comparison uses the identical harness so this convention cancels.

**Command.** `python scripts/run_h2_deeplog.py --rounds 12 --seed 0`  (~228 s on 4 CPU threads)

**Result (`results/h2_deeplog.csv`, seed 0, rounds 12):**

| domain | single-domain F1 | mixed-federation F1 | drop (single-mixed) |
|--------|------------------|---------------------|---------------------|
| HDFS   | 0.7039           | 0.7037              | +0.0002             |
| BGL    | 0.6493           | 0.7518              | -0.1025             |

**Reading — H2 is NOT SUPPORTED (a clean negative).** Mixing does not degrade DeepLog: HDFS
is unchanged (+0.0002) and BGL actually *improves* under mixing. Per CLAUDE.md this is a
publishable result that *strengthens* the reference's generality claim — we report it, we do
not force it positive.

**Why (the mechanism, confirmed independently by H3).** DeepLog feeds the raw event id as a
*scalar* input. In the mixed model HDFS ids live in [0,32] and BGL ids in [33,426] — disjoint
input ranges. The LSTM can separate the domains by input magnitude alone, so weight-averaging
does not blend competing behaviours the way H2 assumed. **Prediction for the paper:** with an
*embedding/one-hot* input (where the two vocabularies overlap in feature space) the
interference should reappear. This is the key follow-up experiment (identified, not yet run).
Figure: `results/h2_deeplog.png`.

**Multi-seed confirmation (3 seeds, `results/h2_multiseed.csv`).** HDFS drop
**+0.0002 ± 0.0001**, BGL drop **−0.1058 ± 0.0023**. The negative is robust: HDFS unchanged
and BGL *improves* under mixing in every seed. (Refactor note: adding the embedding option
did not perturb the scalar path — a scalar smoke reproduces the pre-refactor numbers exactly.)

**Mechanism test built + running.** Added `input_mode='embedding'` to `DeepLog`/`FedConfig`
and an `--embed` flag to both runners. A 1-round smoke already shows the effect reversing sign:
with embedding input, mixed HDFS *drops* +0.094 vs single (scalar showed ~0). Full embedding
H2 + H3 running → `results/h2_deeplog_embedding.csv`, `results/h3_sequential_embedding.csv`.

---

## 5. H3 — sequential arrival (HDFS then BGL)  (2026-07-25)

**What was built.** `scripts/run_h3_sequential.py` + an `init_model` hook on
`train_federated` so DeepLog can be trained on HDFS, snapshotted, then *continue* training on
BGL (the setup where deep models are supposed to forget). Merged vocab (427) fixed throughout.

**Command.** `python scripts/run_h3_sequential.py --rounds 12 --seed 0`  (~112 s)

**Result (`results/h3_sequential.csv`):**

| method       | HDFS F1 after HDFS | HDFS F1 after BGL | forgetting | size after HDFS | size after BGL |
|--------------|--------------------|-------------------|-----------|-----------------|----------------|
| known_events | 0.7885             | 0.7885            | 0.0000    | 1,688 B         | 6,300 B (x3.7) |
| deeplog      | 0.7037             | 0.7037            | 0.0000    | 312,748 B       | 312,748 B      |

**Reading (mixed verdict, split by mechanism).**
- **Set-union: H3 SUPPORTED.** Zero forgetting (union is monotone) but the state grows x3.7 as
  BGL's vocabulary is absorbed — unbounded growth on an open domain stream, exactly the
  bounded-size claim's failure mode under *evolution*.
- **DeepLog: forgetting NOT observed** (HDFS F1 identical to 4 dp after BGL adaptation), while
  size stays fixed. Same root cause as H2: BGL training updates the model in the [33,426] input
  region and barely touches the [0,32] region HDFS occupies. So the deep model is bounded AND
  (here) does not forget — again a scalar-id representational artifact, and again a candidate
  to overturn with an embedding input.

Figure: `results/h3_sequential.png`.

**Net story for the paper.** One reference claim (range invariance) breaks under domain
heterogeneity and we repair it (H1). The other claims (deep-model robustness, and no
forgetting) *survive* for DeepLog — but we show *why*: the scalar-id encoding keeps domains in
disjoint input regions. The bounded-size claim still fails for the set-union family under
sequential arrival. The honest headline is not "everything breaks" but "here is exactly which
property breaks, for which method, and why."

---

## 5b. Mechanism test — embedding input (2026-07-25) — THE DECISIVE RESULT

**Goal.** Test the hypothesis that the H2/H3-DeepLog negatives are caused by the scalar-id
encoding (HDFS ids [0,32] and BGL ids [33,426] disjoint), by swapping the scalar input for a
learned **embedding** (16-dim) where the two vocabularies overlap in feature space, and
re-running H2 and H3.

**Commands.** `python scripts/run_h2_deeplog.py --rounds 12 --seed 0 --embed`;
`python scripts/run_h3_sequential.py --rounds 12 --seed 0 --embed`.
Sources: `results/h2_deeplog_embedding.csv`, `results/h3_sequential_embedding.csv`.
Figure: `results/mechanism_forgetting.png`.

**Result — H3 sequential forgetting (the smoking gun):**

| DeepLog input | HDFS F1 after HDFS | HDFS F1 after BGL | forgetting |
|---------------|--------------------|-------------------|-----------|
| scalar id     | 0.7037             | 0.7037            | **0.0000** |
| embedding     | 0.7037             | 0.0367            | **+0.6670** |

With embedding input, continuing training on BGL **catastrophically forgets HDFS** (F1
0.70 → 0.04). With scalar input, nothing is forgotten. **The mechanism is proven:** the
scalar encoding — not any inherent robustness of the method — is what protected the model.

**Result — H2 simultaneous mixing (an important nuance):**

| DeepLog input | HDFS drop (single−mixed) | BGL drop |
|---------------|--------------------------|----------|
| scalar id     | +0.0002                  | −0.1058  |
| embedding     | −0.0032                  | −0.1198  |

H2 stays **negative even with embeddings**: simultaneous mixed FedAvg does NOT degrade either
domain, regardless of representation. (The 1-round smoke's apparent −0.094 HDFS drop was just
undertraining; it vanishes at 12 rounds.)

**Interpretation (this reshapes the paper).** The dangerous condition is not *mixing* per se
but *sequential arrival with a shared feature space*:
- **Simultaneous mixed FedAvg (H2): robust.** Both domains are present every round, so the
  averaged model maintains both — even when they share an embedding space. This genuinely
  strengthens the reference's generality claim.
- **Sequential arrival (H3): catastrophic for deep models** *iff* the representation makes
  domains overlap (embedding). The scalar encoding hides this by keeping domains in disjoint
  input regions. Set-union, separately, never forgets but grows ×3.7.
So the honest deep-model story is a clean 2×2: {simultaneous, sequential} × {disjoint,
shared-representation}, and only (sequential, shared) fails — but it fails hard (0.70→0.04).

**Multi-seed confirmation (3 seeds, `results/mechanism_multiseed.csv`).** The mechanism is
robust, not a single-seed artifact:
- H3 embedding — DeepLog HDFS forgetting **+0.6647 ± 0.0032** (HDFS F1 after BGL 0.039 ± 0.003).
- H2 embedding — DeepLog HDFS drop **−0.0025 ± 0.0018** (still no degradation under mixing).
- H2 embedding — DeepLog BGL drop **−0.1120 ± 0.0064** (mixed better).
Per-seed H3 forgetting values: 0.6670, 0.6670, 0.6602 — near-deterministic catastrophe.

---

## 7. Ensemble lift — does fixing Length raise Events+Length+ECVC?  (2026-07-25)

**What was built.** A fresh ECVC implementation (`src/methods/ecvc.py`: normalised
event-count histograms, min-L1 nearest-neighbour score, supervised transductive threshold) and
the union ensemble runner (`scripts/run_ensemble.py`). The reference ensemble combines members
by logical OR (`Combine` in the reference repo), so we do the same.

**Finding (honest, and it scopes H1's impact).** On HDFS, ECVC alone is near-saturating
(~0.998 on the capped smoke; full run in `results/ensemble.csv`). Because the ensemble is a
UNION, a collapsed Length cannot lower recall — it simply contributes nothing — and ECVC
already covers the anomalies. Therefore swapping domain-blind Length for domain-aware Length
does **not** change the ensemble's HDFS F1: both sit at the single-domain ensemble level.
Implication: **the domain-aware fix matters for lightweight-only deployments** (Length/Events
without a heavy ECVC nearest-neighbour store), **not for the full ensemble**. This is a precise,
defensible scoping — it does not overclaim that H1 breaks the reference's best method.

---

## 5c. DeepLog reproduction gap — 0.70 vs reference ~0.93 (2026-07-25) — UNRECONCILED, REPORTED

**The gap.** Our single-domain HDFS DeepLog reaches F1 ~0.70; the reference paper reports
DeepLog ~0.91-0.93. This was investigated directly, not hidden.

**Config our runs used** (`FedConfig` defaults + `--rounds 12`): 12 FedAvg rounds x 1 local
epoch, Adam lr=1e-3, batch=1024, CrossEntropy, window=10, hidden=64, layers=2, top-9 rule,
1% train. **No early stopping / no convergence check** — a fixed budget.

**Reference config** (`reference_repo/config_files/hdfs_iid.yaml`): `number_rounds: 1`,
`max_epoch: 1`, `train_per: 0.01`, batch 2048, lr 1e-3, same window/hidden/layers/candidates.
i.e. the reference *federated* DeepLog trains LESS than ours. So under-training on our side is
not the explanation.

**Convergence sweep** (`results/deeplog_convergence_le1_b1024.csv`, full-test eval):
rounds 1/5/12/25/50/100 -> F1 0.087 / 0.704 / 0.704 / 0.698 / 0.671 / 0.721. Plateaus by
round 5; best 0.7205 at 100 rounds. More training does NOT approach 0.93.

**Decision-threshold sweep** (top-g on the plateau model): F1 peaks at g=9 (0.704); recall is
hard-capped at ~0.632 for ALL g<=9 (even g=1). ~37% of abnormal HDFS sequences are predicted
perfectly by the model, so no threshold recovers them.

**Likely reason (stated plainly).** The reference's ~0.93 is most plausibly its CENTRALIZED /
best-tuned DeepLog, not the federated-on-1%-data operating point we (and their federated
config) implement. The prior "0.716" reference run corroborates ~0.70 for the federated regime.
Secondary: raw scalar-id input + 66,802 training windows yield good precision (0.80-0.85) but a
recall ceiling ~0.63 that neither rounds nor g lifts — a representation/data-coverage limit, not
an optimization bug.

**Impact on our claims.** H2/H3 used rounds=12 (F1 0.7039), which sits at the plateau, so those
comparisons already ran at the model's near-best point; the forgetting mechanism (H3 embedding
0.70->0.04) is a *relative* effect and is unaffected by the absolute level. BUT: our absolute
DeepLog numbers are the FEDERATED-1%-DATA operating point (~0.70), NOT the reference's headline
~0.93 — the paper must say this explicitly and not imply we reproduced 0.93. See Threats to
Validity in `PAPER_DRAFT.md` / `docs/paper/main.tex`.

---

## 5d. Centralized capacity-ceiling run (2026-07-26) — CONFIRMS the gap is a method ceiling

To rule out that the ~0.93 gap is a federation artifact or a 1%-subsampling artifact, we ran a
STANDALONE centralized DeepLog on the FULL HDFS normal set (all 558,223 sequences, no FedAvg,
single model), matching every hyperparameter (Adam lr=1e-3, batch=1024, window=10, hidden=64,
2 layers, top-g=9), transductive eval over all 558,223 normal + 16,838 abnormal.
Script: `scripts/deeplog_centralized_hdfs.py` (does NOT touch any federated script/result — it
only imports DeepLog/evaluate read-only). Output: `results/deeplog_centralized_hdfs.csv`.

Run reached round 29/100 (~3.5 h; machine throttled — one round spiked to 69 min) and was then
accepted as converged, because by that point:
- train loss FLATLINED at 0.204 (from 0.349 @ r1; unchanged in the 4th decimal since ~r20);
- F1 PEAKED at 0.4333 @ round 2 (P=0.895, R=0.286), then DECLINED and settled at
  0.2548 +/- 0.0182 over rounds 10-29 (final r29: F1=0.2389, P=0.9749, R=0.1361);
- precision SATURATED at ~0.97-0.98 while recall stayed BELOW 0.30 for every round.

**Finding.** Full data + 100x more passes + no federation does NOT approach 0.93; F1 actually
*peaks early and falls* as the model grows over-confident (precision up, recall down). This is a
hard CAPACITY/METHOD ceiling of the scalar-input next-event DeepLog under the top-g rule on HDFS,
NOT an optimization bug, a federation effect, or a subsampling effect. It corroborates section 5c
directly: the reference's ~0.93 must be a differently-configured (centralized/best-tuned or
different eval population) number. Note the transductive F1 here (~0.25) is on a DIFFERENT eval
population than the held-out ~0.70 federated number and is not directly comparable in level; the
point that carries is the DIRECTION — even the most favourable centralized setting does not lift
DeepLog toward 0.93. Wired into Threats-to-Validity in `docs/paper/main.tex`. Future-work item
"a centralized DeepLog run to pin the reproduction gap" is now DONE and removed from the paper.

---

## 5e. Two publishability upgrades: deployable H1 router + replay fix (2026-07-26)

Motivated by an honest novelty review (H1 alone is near-tautological; the paper diagnosed the
forgetting but never fixed it; the H1 fix assumed an oracle domain tag). Added two REAL
experiments, both in-scope (CPU, HDFS+BGL, no fabrication), turning two "future work" pointers
into results.

**(A) Inferred domain router — closes the H1 oracle-tag hole.**
Script `scripts/h1_inferred_router.py`. The domain-aware Length fix previously routed with the
true domain tag (`[domain]*n`). Replaced it with a router that INFERS domain from the majority
vocabulary block of a sequence's event ids (HDFS ids [0,33), BGL [33,427)), via `vocab.domain_of`.
Result (5 seeds): routing accuracy = **1.0000 +/- 0.0000** on all 569,479 HDFS + 68,874 BGL test
sequences; inferred F1 == oracle F1 EXACTLY (HDFS 0.5614 +/- 0.0236). The disjoint id-blocks that
CAUSE the domain-blind collapse are the same structure that makes the tag recoverable at inference
-> the fix needs no external label. `results/h1_inferred_router.csv`. Wired into main.tex sec:da +
sec:h1.

**(B) Replay mitigation — fixes the (sequential, shared-repr) forgetting.**
Script `scripts/mitigation_replay.py`. Embedding-input DeepLog (the only failing cell). From the
same stage-1 HDFS model, continue on BGL with vs without a replay buffer = p-fraction sample of the
pooled HDFS training set (5,582 seqs), added as one extra federated client. 3 seeds, p in
{0.05,0.1,0.25,0.5}. Model stays fixed-size (buffer stores raw seqs, not params). Results:
  - no-replay : forget +0.6647 +/- 0.0032, HDFS 0.039, BGL 0.642  (reproduces mechanism exactly)
  - p=0.05    : forget +0.4402 +/- 0.3078  (UNRELIABLE — recovers in 1/3 seeds; delimits min buffer)
  - p=0.10    : forget +0.0035 +/- 0.0035, HDFS 0.700 +/- 0.004, BGL 0.722 +/- 0.052  (RELIABLE fix)
  - p=0.25    : forget +0.0013 +/- 0.0009, HDFS 0.702, BGL 0.758  (cleanest)
  - p=0.50    : forget +0.0094 +/- 0.0135, HDFS 0.694, BGL 0.752
**Finding:** a >=10% buffer (~558 seqs) reliably drives forgetting from +0.665 to ~0 while BGL is
learned just as well; 5% too small. Negligible cost, no architecture change, no loss on the new
domain. `results/mitigation_replay.csv`. Wired into main.tex new sec:replay + Table (tab:replay);
abstract/contributions/discussion/future-work updated to diagnose->explain->FIX. EWC left as a
memory-free comparison for future work.

Paper structure re-validated after edits: all \ref resolve, all \cite have \bibitem, 3 figures
present, begin/end balanced, 6 tables.

---

## 6. Status snapshot (2026-07-25)

DONE: dataloader; merged vocab; H1 (5 seeds, IID + non-IID, plot, fix verified); Week-1
baselines; H2 single-seed (negative, plot); H3 (set-union + deeplog, plot); paper draft
skeleton with all final numbers wired. IN FLIGHT: H2 multi-seed (3 seeds). IDENTIFIED NEXT:
embedding-input DeepLog to test whether H2/H3 degradation reappears (the mechanism test).

---

## Appendix — how to reproduce everything above

```bash
# from repo root, with .venv active
# 1. dataloader sanity (vocab sizes, collision check)
PYTHONPATH=. python -m src.vocabulary          # self-test
# 2. H1 single seed
python scripts/run_experiment.py --config configs/hdfs_bgl_mixed.yaml
# 3. H1 multi-seed (writes results/hdfs_bgl_mixed_seed{0..4}.csv + results/h1_multiseed.csv)
python scripts/h1_multiseed.py
# 4. H2 (once torch+flwr installed and deeplog wired) — TODO
```
