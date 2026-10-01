# Impact & Motivation — why this matters to organizations (living research notes)

> Purpose: accumulate the "real-world importance" material that will feed the paper's Introduction,
> Deployment Implications, and the pitch to a professor / reviewer. Every OUR-number here is measured
> or derived from a `results/*.csv` (sourced inline). External industry/regulatory facts are marked
> `[cite]` — to be backed by a real reference before they enter the paper. Nothing here is invented.
>
> Keep appending. Newest context at the bottom of each section.

---

## 1. One-paragraph pitch (the stakes)

Federated log anomaly detection is adopted exactly where centralizing logs is impossible — banks,
healthcare, telecoms, multi-tenant clouds, critical infrastructure. These organizations monitor many
*different* log sources and add new ones continuously. We show that in precisely this setting, the
detectors silently stop working: range-based detection on a tighter domain collapses to near-zero
recall, and a deep model catastrophically forgets an earlier domain the moment a new one is onboarded —
**with no error, no crash, no visible signal.** A monitoring system that has quietly stopped detecting
is more dangerous than one that is down, because operators keep trusting it. The failures are silent,
security-critical, and triggered by routine operations (onboarding a new source). Our fixes cost
almost nothing, and a summary-statistics-only auditor catches the risk *before* deployment.

---

## 2. Where federated log analytics actually runs (and why heterogeneity is the norm)

- **Multi-tenant cloud / MSSP**: one detector federated across tenants whose services (compute,
  storage, network, app, security appliances) emit categorically different logs. Different tenants =
  different domains, onboarded over time.
- **Banking / payments**: core banking, ATM networks, fraud systems, web front-ends — segregated,
  privacy-sensitive, cannot be pooled; regulators require on-prem retention. `[cite PCI-DSS 10.x]`
- **Healthcare**: EHR, imaging, device telemetry across hospitals; HIPAA bars centralizing raw logs.
  `[cite HIPAA Security Rule §164.312(b) audit controls]`
- **Telecom / IoT fleets / smart grid**: millions of edge devices, bandwidth-constrained, new device
  classes added continuously — the canonical "evolving federation."
- **Why it's federated, not centralized**: privacy law, data-residency, bandwidth, and liability make
  shipping raw logs to one place infeasible — the exact motivation of the reference work
  (Pustozerova et al. 2026) and of federated learning generally.

The common thread: **real federations are domain-heterogeneous and evolving** — the regime the two
published claims were never tested in, and the regime where they break.

---

## 3. The silent blind spot — measured (source: `results/impact_blindspot.csv`)

| Failure | Metric | Healthy | Failed | Lost | Operational meaning |
|---|---|---|---|---|---|
| H1 range collapse (HDFS Length, blind mixing) | anomaly recall | 0.3689 | 0.0001 | **99.97%** | **6,209 of 16,838** HDFS anomalies newly missed — silently |
| H3 forgetting (deep model, new domain onboarded) | HDFS detection F1 | 0.7037 | 0.0390 | **94.5%** | detection all but gone the moment a source is added |
| C2 unbounded growth (set-union, per domain) | model state bytes | 1,688 | 6,300 | ×3.7 | memory grows without bound on an open stream |

**Framing for the paper:** the range-collapse number is the sharpest — of the HDFS anomalies the Length
detector *was* catching (6,211), domain-blind mixing silently drops all but ~2. That is a measured,
not rhetorical, "silent blind spot." The forgetting number is a *relative* detection loss (94%),
independent of DeepLog's absolute level (see Threats-to-Validity).

**Security interpretation:** each missed anomaly is a potential intrusion/outage precursor that the
monitoring system no longer surfaces. The onboarding case is worst: *extending* monitoring to a new
service *blinds* monitoring of existing services — scaling coverage destroys coverage.

---

## 4. The cost of the fix — measured (source: `results/impact_overhead.csv`)

| Safeguard | Overhead | Note |
|---|---|---|
| Domain-aware ranges (2 domains) | **32 bytes** | 2 integers per domain |
| Domain-aware ranges (3 domains) | **48 bytes** | scales as O(\|domains\|) |
| Replay buffer (10% of earlier domain) | **≈126 KB** (558 seqs × ~29 events) | raw ids, no parameters |
| DeepLog model (reference point) | ≈347 KB | **the replay buffer is smaller than the model it protects** |
| Pre-deployment auditor (per run) | **0.002 ms** | summary stats only, never raw logs |

**The asymmetry is the headline:** a catastrophic, silent, security-critical failure is prevented by
**32 bytes**, a buffer **smaller than the model**, and a **2-microsecond** pre-deployment check. There
is no efficiency/safety trade-off to agonize over — the safe configuration is essentially free.

---

## 5. Compliance & governance angle (why "silent" is a legal problem, not just technical)

Continuous, *effective* log monitoring is a mandated control in every major regime. A monitoring
system that silently stops detecting is a control failure that an auditor would flag:

- **GDPR Art. 32** — "ability to ensure ongoing confidentiality, integrity... and a process for
  regularly testing, assessing and evaluating the effectiveness" of security measures. `[cite GDPR Art. 32(1)(d)]`
- **HIPAA Security Rule** — audit controls and regular evaluation. `[cite §164.312(b), §164.308(a)(8)]`
- **PCI-DSS v4.0** — Req. 10 (log and monitor), Req. 11 (regularly test security). `[cite PCI-DSS v4.0 Req 10, 11]`
- **SOC 2 / NIST 800-53** — monitoring (AU family) and continuous assessment (CA family). `[cite NIST SP 800-53 AU, CA]`
- **EU NIS2** — operators of essential services must maintain effective detection. `[cite NIS2 Directive Art. 21]`

**The point:** a detector that passes a one-time acceptance test and then silently degrades on the
next onboarding violates the *ongoing-effectiveness* requirement common to all of these. Our
pre-deployment auditor is a concrete way to evidence that the monitoring configuration was checked for
these known failure modes before each change — an auditable artifact.

---

## 6. Threat model (the attacker's view of the blind spot)

- **Blind-spot exploitation:** an adversary active in a domain whose detector has collapsed/forgotten
  operates with the monitoring effectively off, while dashboards show "healthy."
- **Onboarding as the trigger:** the failure is induced by a *legitimate, routine* action (adding a
  log source), so it needs no attacker capability to create — it happens on its own; the attacker only
  needs to be in the right domain afterward.
- **Directionality as a lever:** forgetting is worst when a broad-vocabulary domain is onboarded after
  a narrow one, so the most common "add a big new service" event is the dangerous one.
- **Dwell time:** undetected intrusions persist; industry dwell-time and breach-cost figures quantify
  what a silent blind spot costs. `[cite IBM Cost of a Data Breach; Mandiant M-Trends dwell time]`
  (Use these for the economic framing; do NOT invent a dollar figure.)

---

## 7. From finding to safeguard — the deployable story

- **Pre-deployment auditor** (`fedlog_audit`): from per-client summary statistics alone (event-id
  blocks + length ranges, never raw logs — preserving the federated privacy model) it predicts which
  of these failures a planned federation will hit and prescribes the fix. Validated: **14/15** measured
  configurations (results/auditor_validation.csv).
- **CI/CD gate:** the auditor exits non-zero on HIGH/CRITICAL risk and emits JSON, so a federation
  change can be gated in continuous integration (see `.github/workflows/` and `fedlog-audit --json`).
  This turns a latent *runtime* risk into a *pre-deployment* check — the practical crux for orgs.
- **The fixes themselves** are drop-in: per-domain ranges, tag-free vocab-block routing, a small replay
  buffer (or EWC) — all with the near-zero overhead measured in §4.

---

## 8. Candidate framings / headlines for the paper (draft pool)

- "Scaling coverage can destroy coverage" — the onboarding paradox.
- "A monitoring blind spot is worse than an outage: operators keep trusting it."
- "Catastrophic, silent, and security-critical — prevented by 32 bytes."
- "The safe configuration is essentially free; the only cost is knowing to choose it — which the
  auditor supplies."
- "As log analytics moves to LLM-derived features (which share feature space by construction), this
  failure becomes *more* likely, not less."

---

## 9. To-do for paper integration (tracking)
- [ ] Fold §3/§4 measured numbers into the Deployment Implications section (tables).
- [ ] Add a compact "stakes" paragraph to the Introduction using the 99.97% / 32-byte asymmetry.
- [ ] Back every `[cite]` with a real reference before it enters the paper.
- [ ] Consider a single "impact" figure: blind-spot loss vs fix-cost on a log axis (the asymmetry).
