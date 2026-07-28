# Methods cheatsheet (load on demand)

Lightweight (semi-supervised, train on NORMAL only):
- Length        : learn [min,max] normal length; outside => anomaly. Aggregation: global
                  min/max. H1 TARGET (breaks under domain mixing).
- Known Events  : set of normal event ids; unseen id => anomaly. Aggregation: union.
                  Never forgets, grows unbounded (H3 growth mechanism).
- Edit Distance : min Levenshtein to any known normal seq; > threshold => anomaly.
- N-Gram        : set of length-n subseqs; unseen => anomaly.
- ECVC          : normalised event-count vectors; distance score; > threshold => anomaly.
- Ensembles     : flag if ANY member flags. Best HDFS = Events+Length+ECVC (~0.95).

Deep (semi-supervised): DeepLog, LogAnomaly — LSTM 2x64. Flag if true next event not in
top-g predictions. Aggregation: FedAvg. H2 TARGET (disjoint vocab breaks averaging).

Our fix: DomainAwareLength — per-domain [min,max], routed by domain tag. Open question:
routing when domain is unknown at inference (Week-4 research; measure options, don't guess).
