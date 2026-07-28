# Rules and Goals

## The goal

Produce one IEEE-Access-quality paper showing that federated log anomaly detection
behaves differently under a **heterogeneous, evolving** federation than the literature
reports, and propose an aggregation strategy that fixes the failure.

Success is NOT "our method wins." Success is a defensible, reproducible empirical result
plus (ideally) a working fix. A clean negative result is still a paper.

## The research questions

**Primary RQ.** Do the invariance and bounded-growth properties reported for federated
log anomaly detection survive when the federation is genuinely heterogeneous and its set
of log domains grows over time — and if not, what aggregation strategy restores them?

**H1 (mixing breaks range-based aggregation).**
Length detection learns [min, max] normal sequence length; anything outside is anomalous.
HDFS length is 29±6; BGL is 69±747. Under a mixed federation the global range is
min(all mins)..max(all maxes), which is so wide that HDFS anomalies fall inside it.
Prediction: Length F1 collapses on HDFS in the mixed setting, and the
Events+Length+ECVC ensemble (their best HDFS method) drops with it.
Measure: per-domain F1, single-domain vs mixed.

**H2 (mixing breaks weight averaging).**
DeepLog under FedAvg averages weights across clients. HDFS has 33 event types, BGL has
394; the vocabularies are largely disjoint. Averaging weights over disjoint input spaces
should degrade both domains.
Measure: per-domain F1, single-domain vs mixed, plus FL rounds to converge.

**H3 (sequential arrival is unsustainable for both families).**
Set-union methods never forget (union is monotone) but grow with every new vocabulary;
DeepLog stays fixed-size but forgets earlier domains when adapted to a new one.
Measure: model size after each arrival (bytes), and F1 on domain 1 after domain 2
arrives (forgetting).

## Non-negotiable rules

1. **No fabricated numbers, ever.** Every figure/table cell traces to a file in
   `results/`. Placeholder = `TODO(measure)`, never an invented value.
2. **Reproduce first.** No mixed-domain conclusions before the single-domain baselines
   match the reference paper. The Week-1 gate is real.
3. **Scope = HDFS + BGL.** Two domains. Thunderbird/Spirit = future work.
4. **CPU-only assumptions.** No code that requires a GPU to run at all.
5. **Falsifiable framing.** Each hypothesis has a measurable outcome and a defined
   "what if it's negative" path. We do not move the goalposts to manufacture novelty.
6. **Honest limitations section from day one.** Keep `docs/DECISIONS.md` current so the
   paper's threats-to-validity writes itself.

## Scope cuts already decided (do not re-open without a DECISIONS.md entry)

- Two datasets, not four.
- The "method" contribution is a *simple, well-motivated* domain-aware aggregation
  (e.g. per-domain range parameters + a routing key), not an exotic architecture.
  A simple fix that is clearly explained beats a clever one that misses the deadline.
- Empirical result is the primary contribution; the fix is a bonus.

## What "done" looks like (minimum publishable)

- [ ] Single-domain HDFS + BGL baselines reproduced (Week-1 gate passed).
- [ ] Mixed HDFS+BGL federation run; per-domain F1 reported for lightweight + DeepLog.
- [ ] Sequential arrival run; forgetting + size-growth curves produced.
- [ ] At least the H1 result is clear (positive or negative) with a plot.
- [ ] Domain-aware aggregation implemented and compared (bonus, not blocker).
- [ ] Draft with honest limitations; submit to IEEE Access.
