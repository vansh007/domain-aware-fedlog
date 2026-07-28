# Domain-Aware Federated Log Anomaly Detection under Heterogeneous and Sequentially Arriving Domains

> **Working draft for IEEE Access.** This is a living document. Every number here traces to
> a file in `results/` (cited inline). Sections still awaiting a run are marked
> `TODO(measure)`. Do not submit until every such marker is resolved. Narrative provenance:
> `docs/RESEARCH_LOG.md`; design rationale: `docs/DECISIONS.md`.

**Authors:** Vansh Mundhra et al. (TBD)
**Status:** All experiments complete. H1 (5 seeds, IID + non-IID); H2 (3 seeds, scalar +
embedding); H3 (set-union + DeepLog, scalar + embedding, 3-seed mechanism); ensemble lift done.
Remaining is prose/citations/LaTeX-class only (see end of file). LaTeX: `docs/paper/main.tex`.

---

## Abstract (draft)

Federated log anomaly detection has recently been reported to have two attractive
properties: lightweight statistical detectors are approximately *invariant to client data
distribution*, and their model size is *bounded by a finite event vocabulary*. These
properties, however, were only ever established in a homogeneous setting, where every
federated client holds a slice of the **same** log dataset. We ask whether they survive a
**heterogeneous** federation (clients from different log domains) and an **evolving** one
(new domains arriving over time). Using HDFS and BGL over a single merged 427-event
vocabulary, we show that (H1) domain-blind range-based aggregation collapses on the
tighter-distributed domain — HDFS Length F1 falls from a single-domain **0.538** to
**0.000 ± 0.000** across five seeds — and that a simple **domain-aware** aggregation restores
it to **0.561 ± 0.024**, i.e. back to its single-domain value, while leaving the other domain
untouched. For deep models we find a sharper distinction than prior work anticipates:
*simultaneous* mixed FedAvg does **not** degrade (H2), even with an embedding input, whereas
*sequential* arrival of a new domain **catastrophically forgets** the earlier one (H3: HDFS F1
0.70 → 0.04) — but only when the input representation makes the vocabularies share a feature
space. A controlled experiment (scalar id vs learned embedding) isolates this cause. Set-union
detectors, separately, never forget but grow ×3.7 as one domain arrives, breaking the
bounded-size property under evolution. The net picture is not "everything breaks" but a precise
map — a 2×2 of {simultaneous, sequential} × {disjoint, shared representation} in which exactly
one cell fails — with a simple, effective fix for the aggregation failure that does occur.

---

## 1. Introduction

- Federated learning for log anomaly detection: motivation (privacy, no central log pooling).
- The reference work (Pustozerova et al., IEEE Trans. Reliability, Vol. 75, 2026) benchmarks
  lightweight statistical detectors vs deep models (DeepLog/LogAnomaly) under FedAvg, and
  reports (C1) distribution-invariance of lightweight methods and (C2) bounded model size.
- **Gap.** Their evaluation is *homogeneous*: every client holds a partition of one dataset
  (HDFS *or* BGL), i.e. quantity skew only, never domain skew. Real federations are
  heterogeneous (different services/systems) and evolving (new log sources onboarded over
  time). Their public repo has `hdfs_iid`, `bgl_iid`, ... but **no mixed config**.
- **This paper.** We put HDFS and BGL clients in one federation over a merged vocabulary, and
  also let domains arrive sequentially. We test C1 and C2 directly and propose a minimal
  domain-aware aggregation that repairs the observed failure.
- **Contributions.**
  1. A merged cross-domain vocabulary and dataloader that lets heterogeneous log clients
     share one federation (the enabling artifact the reference pipeline lacks).
  2. An empirical demonstration (5 seeds, IID + non-IID) that domain-blind range aggregation
     collapses on the tighter domain — a clean, mechanistic failure of C1.
  3. A simple domain-aware aggregation that restores the failed component to its
     single-domain performance, with asymmetric, interpretable behaviour.
  4. Analysis of deep FedAvg over disjoint vocabularies (H2) and of both families under
     sequential arrival (H3).

## 2. Background and Related Work

- Log parsing (Drain); event templates; sequence-based anomaly detection.
- Lightweight detectors: Length (range), Known Events (set-union), ECVC, n-grams, edit
  distance. Deep detectors: DeepLog (LSTM next-event prediction), LogAnomaly.
- Federated learning; FedAvg; non-IID/heterogeneity taxonomy (quantity skew vs label/feature/
  **domain** skew). Position our work as the first to test *domain* skew for this task.
- The reference paper's two claims, stated precisely, as our null hypotheses.

## 3. Method

### 3.1 Merged cross-domain vocabulary
- Each domain's parsed event templates form a contiguous ID block: HDFS → global ids [0,33),
  BGL → [33,427). A `domain` tag travels with every sequence. (`src/vocabulary.py`,
  `src/dataloader.py`; rationale `DECISIONS.md` D3.)
- **Event-id collision handled:** HDFS and BGL both reuse small integers for *different*
  templates; we key events as `f"{domain}:{id}"` so `hdfs:5`→0 and `bgl:5`→43 stay distinct.

### 3.2 Federation construction
- `build_federation` assigns each domain N clients; 1% of each domain's normal sequences form
  the training pool (reference convention), partitioned across that domain's clients either
  IID or by log-normal quantity skew (σ=0.25).

### 3.3 Detectors
- **Length (range):** learns [min,max] normal length; flags outside. Domain-blind FedAvg =
  global min-of-mins / max-of-maxes.
- **Known Events (set-union):** learns the set of normal event ids; flags any unseen event.
  FedAvg = union.
- **DeepLog (deep):** 2×64 LSTM predicting the next event id from a length-10 window; a
  sequence is anomalous if the true next event is not in the model's top-9 predictions.
  FedAvg = weight averaging over clients. (`src/methods/deeplog_fed.py`; the LSTM is a fresh,
  minimal re-implementation to keep our code off the GPL reference — `DECISIONS.md` D2.)

### 3.4 Domain-aware aggregation (our fix)
- Keep **per-domain** range parameters, aggregated only across clients of the *same* domain,
  and route a test sequence to its domain's range via the domain tag. (`src/aggregation.py`
  `DomainAwareLength`.) Preserves the federation while removing cross-domain contamination.
- The harder case — routing when the domain tag is unknown at inference — is stated as an open
  problem (`predict_domain_unknown`), not silently resolved.

## 4. Experimental Setup

- Datasets: LogHub HDFS (33 event types; 558,223 normal / 16,838 abnormal sequences) and BGL
  (394 event types; 37,823 normal / 31,429 abnormal). Merged vocab = 427.
- Federation: 3 HDFS + 2 BGL clients, merged 427 vocabulary, IID unless stated.
- Seeds: 0–4; we report mean ± population std. CPU only.
- Metrics: per-domain precision/recall/F1 (anomaly = positive); forgetting = F1_before −
  F1_after; model state size in bytes.
- Reproduction gate (Table 1): single-domain baselines match the reference — HDFS Length
  0.538 (ref ~0.57), BGL Known Events 0.992 (ref ~0.99). (`results/{hdfs,bgl}_baseline.csv`.)

## 5. Results

### 5.1 H1 — range aggregation collapses under domain mixing, and the fix restores it

**Setup.** Mixed 3 HDFS + 2 BGL federation, merged 427 vocab, 5 seeds.
Source: `results/h1_multiseed.csv`, `results/h1_multiseed_lognormal.csv`. Figure:
`results/h1_collapse_recover.png`.

**Table 2 — per-domain F1 (mean ± std, n=5), IID.**

| Method                       | HDFS F1         | BGL F1          |
|------------------------------|-----------------|-----------------|
| Length, global (domain-blind)| 0.0000 ± 0.0001 | 0.0332 ± 0.0435 |
| Length, domain-aware (ours)  | 0.5614 ± 0.0236 | 0.0332 ± 0.0435 |
| Known Events (set-union)     | 0.5636 ± 0.1170 | 0.9866 ± 0.0034 |
| *HDFS single-domain Length (ref line)* | *0.5379* | — |

**Findings.**
1. **Collapse.** Domain-blind Length on HDFS is F1 ≈ 0.000 — BGL's huge length spread widens
   the shared global range until almost no HDFS sequence looks anomalous. C1 (distribution-
   invariance) fails under domain skew.
2. **Recovery.** Domain-aware Length restores HDFS to 0.561 ± 0.024 — statistically
   indistinguishable from its single-domain baseline (0.538). The fix returns the component
   to its standalone usefulness.
3. **Asymmetry.** Length on BGL is *identical* under both aggregations in every seed: BGL is
   the domain that *widens* the range, not its victim. Damage and repair are both one-sided.
4. **Specificity (contrast case).** Known Events (set-union) is unaffected by mixing — not
   every lightweight method breaks, only the range-based one. This isolates the mechanism.
5. **Non-IID.** Under log-normal quantity skew the story is unchanged (HDFS Length 0.0001 →
   0.5614); domain-aware Length is *partition-invariant* by construction (per-domain min/max
   is independent of how a domain's data is split across clients).

**Honest scope.** 0.561 is a recovery of the *Length component*, not the ~0.95 of the full
Events+Length+ECVC ensemble. Whether repairing the component lifts the whole ensemble is the
subject of §5.4 / future work.

### 5.2 H2 — DeepLog under FedAvg over (near-)disjoint vocabularies

**Setup.** Single-domain vs mixed FedAvg DeepLog (2×64 LSTM, window 10, top-9 rule, 12 FedAvg
rounds), per-domain F1. Source: `results/h2_deeplog.csv`. Figure: `results/h2_deeplog.png`.

**Table 3 — DeepLog per-domain F1, single-domain vs mixed (mean ± std, 3 seeds).**
Source: `results/h2_multiseed.csv`.

| Domain | Single-domain F1 | Mixed-federation F1 | Drop (single − mixed) |
|--------|------------------|---------------------|-----------------------|
| HDFS   | 0.7039           | 0.7037              | **+0.0002 ± 0.0001**  |
| BGL    | 0.6459           | 0.7517              | **−0.1058 ± 0.0023**  |

The drop is essentially zero on HDFS and consistently *negative* (mixed better) on BGL across
all three seeds — the negative is robust, not a single-seed accident.

**Finding — H2 is NOT supported (clean negative).** Mixing does not degrade DeepLog: HDFS is
unchanged and BGL *improves* under mixing. This *strengthens* the reference's generality claim
for the deep model, and we report it as such rather than forcing a positive result.

**Robust even under an embedding input.** We re-ran H2 with a learned 16-dim embedding
(vocabularies overlap in feature space): HDFS drop −0.0032, BGL drop −0.1198 — still no
degradation (`results/h2_deeplog_embedding.csv`). **Simultaneous mixed FedAvg is robust
regardless of representation**, because every round sees both domains and the averaged model
maintains both. This is a genuine strengthening of the reference's generality claim — but note
(§5.3) that it does *not* extend to *sequential* arrival.

### 5.3 H3 — sequential arrival: grow-forever vs forget

**Setup.** HDFS arrives, then BGL; measure HDFS forgetting and model size after each arrival.
Source: `results/h3_sequential.csv`. Figure: `results/h3_sequential.png`.

**Set-union (Known Events).** HDFS F1 after HDFS = 0.7885; after BGL = 0.7885 → **forgetting
= 0.0000** (union is monotone). Model state grows **1688 → 6300 bytes (×3.7)** as BGL's
vocabulary is absorbed — unbounded growth on an open stream of domains.

**DeepLog (scalar input).** Trained on HDFS, then *continued* on BGL: HDFS F1 0.7037 → 0.7037
→ **forgetting = 0.0000**, size fixed. No forgetting — BGL training updates the [33,426] input
region and barely perturbs the [0,32] region HDFS occupies.

**Table 4 — H3 sequential arrival (HDFS → BGL, seed 0).**

| Method / input          | HDFS F1 after HDFS | HDFS F1 after BGL | Forgetting | Size after HDFS | Size after BGL |
|-------------------------|--------------------|-------------------|-----------|-----------------|----------------|
| Known Events            | 0.7885             | 0.7885            | 0.0000    | 1,688 B         | 6,300 B (×3.7) |
| DeepLog (scalar id)     | 0.7037             | 0.7037            | 0.0000    | 312,748 B       | 312,748 B      |
| **DeepLog (embedding)** | 0.7037             | **0.0367**        | **+0.6670** | 355,436 B     | 355,436 B      |

**Takeaway (split by mechanism).** Set-union: *bounded-size* claim fails under evolution —
never forgets but grows ×3.7, unbounded on an open stream. DeepLog: the forgetting depends
entirely on the input representation (next section).

### 5.4 Mechanism test — the decisive result (DONE)

We swapped DeepLog's scalar id input for a learned 16-dim **embedding**, where the two
vocabularies overlap in feature space, and re-ran H2 and H3. Figure:
`results/mechanism_forgetting.png`.

- **H3 sequential arrival, embedding input (3 seeds):** continuing training on BGL
  **catastrophically forgets HDFS — forgetting +0.6647 ± 0.0032** (HDFS F1 0.7037 →
  0.039 ± 0.003), versus 0.000 with scalar input. This *proves* the mechanism: the scalar
  encoding, not any inherent robustness, is what protected the model; once the domains share a
  feature space, adaptation destroys the earlier domain.
- **H2 simultaneous mixing, embedding input (3 seeds):** still no degradation (HDFS drop
  −0.0025 ± 0.0018, BGL −0.1120 ± 0.0064). Simultaneous FedAvg is robust regardless of
  representation. Source: `results/mechanism_multiseed.csv`.

**The unifying picture is a 2×2:** {simultaneous mixing, sequential arrival} ×
{disjoint-range input, shared-representation input}. Only **(sequential, shared
representation)** fails — and it fails hard (0.70 → 0.04). *Simultaneous mixing is safe; it is
sequential onboarding of a domain into a shared-representation model that is dangerous.* This
distinction — not present in the reference paper — is a central contribution.

### 5.5 Does the fix lift the full Events+Length+ECVC ensemble? (DONE — it does not, and that
is the honest, correctly-scoped answer)

Source: `results/ensemble.csv` (our fresh ECVC in `src/methods/ecvc.py`; union combination as
in the reference). We first reproduce the reference's strong HDFS ensemble: **single-domain
Events+Length+ECVC F1 = 0.984** (reference reports ~0.95). Then, in the mixed federation:

| HDFS ensemble                     | F1     |
|-----------------------------------|--------|
| single-domain (reference target)  | 0.9836 |
| mixed, Length **global**          | 0.9837 |
| mixed, Length **domain-aware**    | 0.9836 |

**The ensemble F1 is unchanged** whether Length is global (collapsed, 0.0002) or domain-aware
(recovered, 0.5379). Reason: the ensemble is a logical **OR**, and on HDFS **ECVC alone reaches
0.968** with recall ~0.999; a collapsed Length cannot lower a union's recall, and a recovered
Length adds only detections ECVC already makes. So Length's contribution inside this ensemble is
redundant.

**Correct scoping of H1's impact (not an overclaim).** The domain-aware fix is decisive for
**lightweight-only** deployments — Length/Events without a heavy ECVC nearest-neighbour store,
which is exactly the regime the reference motivates on cost grounds — where HDFS Length goes
from 0.000 to 0.561. It does **not** change the full ensemble, because the ensemble is carried by
ECVC. We report this rather than claim the mixed federation breaks the reference's best method.

## 6. Threats to Validity
- **DeepLog absolute level is the federated-1%-data operating point (~0.70), not the
  reference's headline ~0.93.** We state this explicitly. Our single-domain HDFS DeepLog
  plateaus at F1 ~0.70 (best 0.7205 over a 1–100 round sweep; `results/deeplog_convergence_*`),
  and neither more training nor top-g threshold tuning reaches 0.93 (recall is hard-capped at
  ~0.63). The reference's *federated* config itself trains only 1 round × 1 epoch on 1% data, so
  our number is consistent with its federated regime; the ~0.93 figure is most plausibly the
  reference's centralized/best-tuned DeepLog. We therefore do **not** claim to reproduce 0.93,
  and all H2/H3 DeepLog results should be read as the federated-1%-data regime. The H3 forgetting
  finding is a *relative* effect (0.70→0.04 under embedding vs 0.70→0.70 under scalar) and does
  not depend on the absolute level; H2/H3 were run at rounds=12, which is already at the plateau.
- **Two domains only** (HDFS, BGL). Two genuinely different domains already exhibit the
  failure; more would widen the predicted gap (`DECISIONS.md` D1).
- **DeepLog input representation** (scalar id) affects the H2 magnitude — discussed in §5.2.
- **Single method, not full ensemble** for the reproduction gate; ensemble lift is future work.
- **Known Events variance on HDFS** (0.56 ± 0.12) is high across seeds — reported, not hidden.
- **Eval convention:** sequences shorter than the window are treated as normal by our DeepLog
  eval; the single-vs-mixed comparison uses the identical harness, so the relative effect is
  unaffected.

## 7. Conclusion
- The reported invariance property does not survive domain heterogeneity for **range-based
  aggregation** (H1); a minimal **domain-aware** fix restores it to single-domain performance.
- For **deep models**, simultaneous mixed FedAvg is robust (H2, even with embeddings), but
  **sequential** onboarding of a domain into a shared-representation model causes catastrophic
  forgetting (H3, 0.70 → 0.04) — a danger the reference's scalar-id encoding happens to mask.
- **Set-union** detectors never forget but grow unbounded (×3.7 for one added domain).
- The contribution is a precise, reproducible map of which property fails, for which method,
  under which arrival pattern and representation — plus a working fix for the aggregation case.
  Simple, honest, reproducible; every number traces to `results/`.

---

### Figure/table manifest (all from `results/`)
- Table 1 (repro gate): `hdfs_baseline.csv`, `bgl_baseline.csv`
- Table 2 / Fig H1: `h1_multiseed.csv`, `h1_multiseed_lognormal.csv`, `h1_collapse_recover.png`,
  `h1_collapse_recover_lognormal.png`
- Table 3 / Fig H2: `h2_multiseed.csv` (+ per-seed `h2_deeplog_seed{0,1,2}.csv`),
  `h2_deeplog_embedding.csv`, `h2_deeplog.png`
- Table 4 / Fig H3: `h3_sequential.csv`, `h3_sequential_embedding.csv`, `h3_sequential.png`
- Mechanism: `mechanism_forgetting.png`, `mechanism_multiseed.csv` (3-seed),
  per-seed `h{2,3}_*_embedding_seed{0,1,2}.csv`
- §5.5 ensemble: `ensemble.csv` (ECVC = `src/methods/ecvc.py`)
- LaTeX source: `docs/paper/main.tex` (+ `docs/paper/README.md`)

### Remaining before submission
- Author list, affiliations, and citations (reference paper, DeepLog, ECVC/ADSD, LogHub, Drain,
  FedAvg).
- Switch LaTeX to the official IEEE Access class (`ieeeaccess.cls`); compile (no LaTeX in this
  environment).
- Optional deeper dives: domain-unknown routing for domain-aware Length (the open Week-4
  question in `src/aggregation.py`); embedding-DeepLog with a continual-learning mitigation
  (e.g. replay) to complete the H3 story.
