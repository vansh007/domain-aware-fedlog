# Reference Paper — distilled notes

> Purpose: so no future session has to re-read the full PDF. If a claim here is
> load-bearing for the paper, verify it against the source before publishing.

**Paper:** Pustozerova, García Gómez, Landauer, Wurzenberger, Skopik, Weippl, Mayer,
Ekelhart. "Lightweight Techniques for Federated Anomaly Detection in Log Data."
IEEE Transactions on Reliability, Vol. 75, 2026. DOI 10.1109/TR.2026.3677674.
**Code:** https://github.com/ait-aecid/comparison-fed-centr-efficient-ad (GPL-3.0, Flower, Python 3.12.3)

## What they built
A federated log anomaly detection pipeline comparing lightweight statistical methods
against LSTM deep learning (DeepLog, LogAnomaly), on HDFS and BGL, using Flower.
They release code. They evaluate 5/10/50 clients, IID and non-IID.

## Their pipeline steps (their Fig. 1)
1. Data collection (logs stay local)
2. Parser execution (Drain) -> templates
3. ID/template synchronisation -> global dictionary D = {(template, event_id)}
4. Data association -> each client maps its logs to global event IDs -> event sequences
5. Local training
6. Aggregation
7. Parameter update (redistribute global model)
8. Evaluation
Note: they put federated *parsing* (steps 2-4) OUT of scope and assume clients already
have data parsed into global templates. We inherit that assumption for now.

## The lightweight methods (semi-supervised; trained on normal only)
- **Length**: learn [min, max] normal sequence length; outside range => anomaly.
  Federated: global min = min of local mins, global max = max of local maxes.  <-- H1 target
- **Known Events**: set of all normal event IDs; unseen event => anomaly.
  Federated: union of local sets.
- **Edit Distance**: min Levenshtein distance to any known normal sequence; > threshold => anomaly.
- **N-Gram**: set of length-n subsequences; unseen subsequence => anomaly.
- **ECVC** (Event Count Vector Clustering): normalised count vectors; distance-based score.
- **Combinations**: 2-Gram+Length; Events+Length; Events+Length+ECVC; Events+Length+Edit.
  Ensemble = anomaly if ANY member flags it.

## Deep baselines
- **DeepLog**, **LogAnomaly**: LSTM, 2 hidden layers x 64 units. Semi-supervised
  (train on normal; flag if true next event not in top-g predictions).
- Aggregated with **FedAvg**.  <-- H2 target

## Dataset facts (their Section V-A)
- **HDFS**: 33 event types; 558,223 normal + 16,838 abnormal sequences; avg length 29±6.
- **BGL**: 394 event types; 37,823 normal + 31,429 abnormal sequences; avg length 69±747.
- Training uses 1% of normal sequences (standard in this literature); rest for eval.

## Results we must reproduce (their Table II, approx)
- HDFS: ECVC and Events+Length+ECVC ~0.95 F1, beating DL (~0.91-0.93, p<0.05).
  Length alone ~0.57, EditDistance ~0.72 (poor on HDFS).
- BGL: Events and Events+Length+Edit ~0.99 (BGL anomalies often contain unseen events).
  Length and n-grams poor on BGL.
- Lightweight methods need only **1 aggregation round**; DL needs several (more with 50 clients).

## The two claims we are challenging
- **Claim 1 (invariance):** lightweight global models are built by deterministic
  set-union / min-max, "independent of client data distributions" => identical F1 in
  centralised, IID, and non-IID. TRUE within one dataset. Untested across datasets.
- **Claim 2 (bounded growth):** model size bounded by finite event vocabulary; "once
  the vocabulary is saturated, memory usage stabilises." Their Table V already shows
  Edit and ECVC reaching DL model sizes on a single dataset.

## The gap (critical)
Their "non-IID" = quantity skew via log-normal(mu=0, sigma=0.25) allocation of training
data amounts. Every client still holds the SAME dataset. HDFS and BGL are evaluated in
separate experiments — never in one federation. Their repo has hdfs_*.yaml and
bgl_*.yaml but no mixed config.

## Known repo issues (save yourself the debugging)
- README references `ml_centralize.py`; actual file is `ml_centralised.py`.
- `simulation_app.py` finishes with a Flower exception AFTER saving results (documented
  upstream; not fatal — results are written before the crash).
- Repo has 1 star, 0 forks => essentially untested outside the authors' lab. Expect
  rough edges. Budget debugging time in Week 1.

## Their own threats-to-validity (useful for our paper)
- They trained multiple clients on a single machine (affects timing, not F1).
- Cross-study metric comparison is unreliable; they only compare within their codebase
  with controlled splits and averaged runs. We adopt the same discipline.
