# Review 2 Prep — Everything, In Plain English

> **Read this once end-to-end and you will understand the whole project.**
> It is written for a person who is NOT already deep in the code. It explains
> the jargon, tells you what we actually built and found, and maps everything
> onto the 7 things Review 2 grades you on. There are also copy-paste prompts
> for ChatGPT/DALL·E to generate the diagrams you'll need.
>
> Golden rule for the viva: **every number we report came from a real experiment
> we ran** (they live in the `results/` folder as CSV files). We never made a
> number up. If a panelist asks "where's that number from," the honest answer is
> always "from a run, here is the file."

---

## PART 0 — The 30-second version (memorize this)

> "System logs are the diaries that computers write. We use AI to spot the 'bad'
> entries that signal a failure or attack. Normally you'd collect everyone's logs
> in one place to train the AI — but that's a privacy and bandwidth nightmare.
> **Federated learning** avoids it: each computer trains locally and only shares
> the *model*, never the raw logs.
>
> A recent 2026 paper claimed two nice things about the *cheap* (lightweight)
> methods for doing this. But they only ever tested it where **every computer had
> the same kind of logs**. We asked the obvious next question: **what if the
> computers have *different* kinds of logs, and what if new kinds show up over
> time?** — because that's what the real world looks like.
>
> We found that **some of their claims break** in that realistic setting, we
> explained *exactly why* they break, and we built **simple fixes that repair
> them**. That gap — a missing 'mixed' experiment — is our entire contribution."

That's the whole thesis. Everything below is detail.

---

## PART 1 — The big picture, told as a story

### 1.1 What is a "log" and why do we care?

Every server, app, and device constantly writes tiny status messages to a file —
"user connected," "block replicated," "disk read failed." That file is a **log**.
When something goes wrong (a crash, a hack, an outage), the log is the black-box
recorder engineers use to figure out what happened.

Because there are *millions* of log lines, humans can't read them all. So we train
software to automatically flag the **anomalous** (abnormal / suspicious) sequences.
That's **log anomaly detection**.

### 1.2 The privacy problem, and the federated answer

The classic way to train a good detector is to gather everyone's logs into one big
pile and learn from it (**centralized** learning). But logs are sensitive (they can
contain user activity, internal system details) and huge (expensive to ship around).
Hospitals, banks, IoT fleets often *legally cannot* pool their logs.

**Federated learning (FL)** is the fix. Analogy: instead of every student mailing
their private notebook to one teacher, each student studies at home and only mails
back *"here's what I learned"* (a small model update). A coordinator averages those
updates into one shared model. **Raw data never leaves the device.** The most common
recipe for this averaging is called **FedAvg** (Federated Averaging) — literally,
average everyone's model weights together.

### 1.3 The paper we are building on

A 2026 IEEE paper (Pustozerova et al.) compared two families of federated log
detectors:

- **Lightweight / statistical detectors** — dead simple, cheap, no neural network.
  E.g. "normal log sequences are between 20 and 40 events long; flag anything
  outside that range." Tiny and fast.
- **Deep detectors** — a small neural network (**DeepLog**) that learns the *grammar*
  of normal logs and flags anything that doesn't fit.

They reported **two attractive claims** about the lightweight ones:
- **Claim C1 (invariance):** these methods don't care how the data is split across
  clients — the federated result equals the centralized result.
- **Claim C2 (bounded size):** the model can't grow forever, because there's only a
  finite list of possible log events (a finite **vocabulary**).

### 1.4 The gap we exploit (this is the heart of the project)

Here's the catch. **They only ever tested this where every client held the *same*
dataset.** Their public code ships config files `hdfs_iid`, `hdfs_no_iid`,
`bgl_iid`, `bgl_no_iid` — but **there is NO `hdfs_bgl_mixed`**. Nobody ever put
*two different kinds of logs in one federation.*

> **That single missing config file is the entire paper.** — from `CLAUDE.md`

Real federations are **heterogeneous** (a bank's logs look nothing like a smart
grid's) and **evolving** (new log sources get onboarded over time). So we test the
two claims in exactly the setting they were never tested in:

1. **Mixed / heterogeneous:** put HDFS logs and BGL logs in *one* federation at the
   same time.
2. **Sequential / evolving:** start with HDFS, then a while later BGL "arrives."

### 1.5 What we found (the punchline)

We organized findings as three hypotheses. Short version:

- **H1 — the cheap "length range" method BREAKS when domains are mixed.** BGL logs
  vary wildly in length, HDFS logs are tight. When you naively merge their "normal
  length range," the combined range gets so wide that *no HDFS anomaly ever looks
  abnormal anymore.* HDFS detection score (F1) crashes from **0.54 → 0.00**.
  **Our fix:** keep a *separate* range per domain instead of one global range. That
  restores HDFS back to **0.56** — its original level — without hurting BGL. And the
  fix needs **no manual label**: we can *infer* which domain a log came from with
  **100% accuracy**, so it's deployable for real.

- **H2 — the deep model does NOT break when domains are mixed *simultaneously*.**
  This is a *negative* result, and that's fine — it actually *strengthens* the deep
  method's reputation. Reason: HDFS and BGL events use non-overlapping ID numbers, so
  the network keeps them separate automatically.

- **H3 — but when a new domain arrives *later* (sequentially), the deep model
  suffers CATASTROPHIC FORGETTING** — it learns BGL and *forgets* HDFS (HDFS F1
  **0.70 → 0.04**). AND we proved *why*: it only forgets when the input
  **representation** makes the two vocabularies share the same "feature space." With
  the simple representation, no forgetting. Meanwhile the cheap set-based method never
  forgets but its size **grows ×3.7** — breaking claim C2. **Our fix:** keep a tiny
  **replay buffer** (~10% of old data, a few hundred sequences) and the forgetting
  drops from 0.665 → 0.004.

The elegant result is a clean **2×2 map**: *{mixed-at-once, arrives-later} ×
{separate-number-ranges, shared-feature-space}* — and **only ONE of the four cells
fails** (arrives-later + shared-feature-space). We're the first to draw that map.

> **We did not force a positive result.** H2 came back negative and we report it
> honestly — a negative that strengthens the prior work is still a publishable
> finding. That intellectual honesty is a strength to emphasize in the viva.

---

## PART 2 — The scary terminology, decoded (with how WE use each)

Keep this table open during the viva. For each term: the plain meaning, then
*specifically how it shows up in our project.*

| Term | Plain-English meaning | How it's used in OUR project |
|---|---|---|
| **Log / log sequence** | The status messages a computer writes; a "sequence" is one session's worth of them. | Our input data. Each sequence is a list of event IDs, labelled normal or abnormal. |
| **Anomaly detection** | Automatically flagging the "abnormal" items. | The task. We flag abnormal log sequences. |
| **HDFS** | A benchmark log dataset from a Hadoop distributed file system. 33 event types, ~558k normal + ~16.8k abnormal sequences, short & tight (avg length 29±6). | Domain #1. The "tight-length" domain, and the *victim* in H1. |
| **BGL** | A benchmark log dataset from a Blue Gene/L supercomputer. 394 event types, ~37.8k normal + ~31.4k abnormal, wildly variable length (avg 69±**747**). | Domain #2. The "wide-length" domain that *contaminates* the shared range in H1. |
| **Domain** | A *kind* of log — one generating system. HDFS and BGL are two domains. | The variable we vary. Prior work = one domain per experiment. Us = two domains together. |
| **Heterogeneous federation** | A federation where clients hold *different kinds* of data. | Our core novel setting: HDFS clients + BGL clients in one federation. |
| **Federated learning (FL)** | Train locally, share only model updates, never raw data. | The privacy-preserving framework we operate inside. |
| **FedAvg** | The standard FL recipe: average all clients' model weights. | How we combine client models for both the deep and lightweight methods. |
| **Client** | One participant device holding its own local data. | We use 3 HDFS clients + 2 BGL clients by default. |
| **IID / non-IID** | IID = every client's data looks statistically the same. non-IID = it doesn't (the hard, realistic case). | We test both. "**Quantity skew**" (clients differ only in *amount* of data) is what the prior paper tested. |
| **Domain skew / feature skew** | A *severe* kind of non-IID where clients hold data from different generating processes (different features entirely). | **This is our regime** — the extreme the prior work never tested. Sits at the far end of "feature skew." |
| **Vocabulary** | The finite set of distinct event types that can appear. | HDFS has 33, BGL has 394. We **merge** them into one 427-event global vocabulary. |
| **Merged / global vocabulary** | One combined event dictionary spanning both domains. | Our key enabling artifact. HDFS gets IDs 0–32, BGL gets IDs 33–426. This is `src/dataloader.py`. |
| **DeepLog** | A small 2-layer LSTM neural net that predicts the next log event; flags a sequence if the real next event isn't in its top-9 guesses. | Our "deep model." The subject of H2 and H3. |
| **LSTM** | A type of neural network good at learning *sequences* (remembers order). | The engine inside DeepLog. |
| **Scalar-ID input vs. Embedding input** | Two ways to feed an event to the network. Scalar = just its raw ID number. Embedding = a learned 16-number vector representing the event's "meaning." | **The single most important variable in our paper.** Scalar keeps domains in separate number ranges (safe). Embedding makes them share a feature space (forgets). |
| **Feature space** | The mathematical space where the model "thinks." Two things sharing a feature space compete for the same neurons. | With embeddings, HDFS and BGL share it → they interfere → forgetting. With scalar IDs, they don't → no forgetting. |
| **Length detector** | A lightweight method: learn the [min, max] length of normal sequences, flag anything outside. | The star of H1 — the method that collapses under mixing and that we fix. |
| **Known-Events / set-union detector** | Lightweight: remember the *set* of events ever seen as normal; flag any never-before-seen event. Merging = set union. | The "grows forever" method in H3 — never forgets but size balloons ×3.7. |
| **ECVC** | A lightweight method comparing a sequence's event-count histogram to the nearest normal one. | An ensemble member; explains why our H1 fix doesn't change the *full* ensemble (ECVC alone is already strong). |
| **Ensemble** | Combine several detectors (logical OR: flag if *any* fires). | The prior paper's strongest HDFS method (Events OR Length OR ECVC). |
| **Catastrophic forgetting** | When a neural net learns a new task and *abruptly forgets* an old one. | The H3 failure: learn BGL, forget HDFS (0.70 → 0.04). |
| **Continual / lifelong learning** | The field studying how to learn new things without forgetting old ones. | Sequential domain arrival IS a continual-learning problem; forgetting is its central villain. |
| **Experience replay** | Anti-forgetting trick: keep a small buffer of old examples and re-train on them alongside new data. | **Our H3 fix.** A ~10% HDFS buffer kills the forgetting. |
| **EWC (Elastic Weight Consolidation)** | Another anti-forgetting trick that protects "important" weights instead of storing data. | We *cite* it as future-work alternative to replay; we didn't run it (honest scoping). |
| **Precision / Recall / F1** | Precision = of the things we flagged, how many were truly bad. Recall = of the truly bad things, how many we caught. F1 = their balance (0=worst, 1=best). | Our main metric. "F1 = 0.56" means a decent balanced detector; "F1 = 0.00" means broken. |
| **Seed** | A random-number starting point. Re-running with different seeds checks a result isn't a fluke. | We report **mean ± std over 3–5 seeds** so results are trustworthy, not luck. |
| **Transductive vs. held-out evaluation** | Transductive = tuned/tested on the same set. Held-out = tested on unseen data. | A nuance in our threats-to-validity; explains a number difference honestly. |
| **Forgetting (metric)** | F1-before-new-domain minus F1-after. Big positive = lots forgotten. | Our H3 headline number: +0.665 (bad) → +0.004 with replay (fixed). |

---

## PART 3 — Mapping onto the 7 Review-2 parameters

This is the section to actually build your report/slides from. Each sub-section
tells you *what the panel wants to hear* and *our exact content to say*.

### ① Domain understanding & problem definition (3 marks) — CO2

**What they want:** you clearly understand the field, and the problem is relevant,
well-scoped, with stated assumptions/constraints.

**Say this:**
- **Domain:** federated (privacy-preserving) anomaly detection on system logs.
- **The problem:** a 2026 IEEE paper made two reassuring claims about lightweight
  federated log detectors, but validated them only in a *homogeneous* setting
  (every client, same dataset). Real federations are **heterogeneous** (different
  log domains) and **evolving** (new domains over time). **Do the claims still hold
  there?**
- **Why it matters:** if a claimed "invariant, bounded" method silently breaks when
  a second kind of log joins the federation, practitioners deploying it get a false
  sense of safety. We turn that unknown into a precise, tested map.
- **Assumptions/constraints (state them — examiners love this):**
  - Two domains only (HDFS, BGL) — adding more needs new parsers = out of scope.
  - CPU-only, laptop-scale (the deep model is deliberately tiny).
  - Semi-supervised: we train on *normal* sequences only (the realistic assumption —
    you rarely have labelled failures up front).
  - We reproduce the reference's single-domain numbers *first* before trusting any
    mixed-domain conclusion (a self-imposed correctness gate).

### ② Literature / patent review & analysis of existing approaches (3 marks) — CO2

**What they want:** you surveyed credible sources, compared approaches, and found
the *gap*.

**Say this — the landscape has 4 buckets, and our gap sits between them:**

| Existing work | What it does | Limitation we exploit |
|---|---|---|
| **Pustozerova et al. 2026** (our anchor) | Lightweight vs deep federated log detectors; claims C1 (invariance) + C2 (bounded size). | Only tested one dataset per federation. Never mixed domains. **← our gap.** |
| **DeepLog (2017), LogAnomaly (2019)** | Deep log anomaly detectors. | Not federated; not tested cross-domain. We reuse DeepLog's mechanism. |
| **FedAvg (2017), non-IID FL (Zhao 2018), FL survey (Kairouz 2021)** | Federated averaging + data-heterogeneity theory (quantity/label/feature skew). | "non-IID hurts FedAvg" is the usual story. **Domain skew (extreme feature skew) for logs is under-explored** — and our H2 is a *partial counterpoint*. |
| **Continual/federated-continual (EWC 2017, De Lange 2022, Dupuy 2023)** | Catastrophic forgetting & mitigations. | Nobody identified the **input representation** as the switch that decides whether forgetting happens for federated logs. |
| **FedLAD (2025), federated-deep-log open challenges (2024)** | Testbeds/infrastructure for federated log AD. | Provide tooling to *vary* clients but **never isolate the two claims under a single mixed-domain federation.** |

**The one-line gap:** *No prior work isolates the invariance/bounded-size claims
under a single federation that mixes two distinct log domains, or under sequential
domain arrival, and none names the input representation as the lever governing
forgetting.* (This is literally our Related Work paragraph.)

### ③ Objectives, scope & expected outcomes (2 marks) — CO2, CO4

**Objectives (specific & achievable):**
1. Build a merged cross-domain vocabulary + dataloader so heterogeneous log clients
   can share one federation (the missing artifact).
2. **H1:** test whether range-based lightweight aggregation survives domain mixing;
   if not, fix it.
3. **H2:** test whether deep FedAvg survives *simultaneous* domain mixing.
4. **H3:** test whether lightweight/deep methods survive *sequential* domain arrival;
   if they forget/grow, mitigate it.

**Scope (in vs out):**
- **In:** HDFS + BGL; CPU; FedAvg; lightweight (Length/Known-Events/ECVC) + DeepLog;
  IID and non-IID (quantity skew).
- **Out:** more than two domains, Thunderbird/Spirit datasets, GPU-scale models,
  new FL algorithms beyond FedAvg. (Future work.)

**Expected outcomes (what a "done" looks like):**
- A 2×2 map of when federated log AD is safe vs unsafe.
- Two working fixes (domain-aware aggregation; replay buffer).
- A reproducible CPU artifact where every number traces to a CSV.
- A paper targeted at **IEEE Access**.

### ④ Proposed methodology (3 marks) — CO2, CO4

**What they want:** appropriate methods, algorithms, data, tools, and a sound
experimental approach.

**The method, step by step:**
1. **Data prep:** parse HDFS & BGL raw logs into event-ID sequences (using the
   standard Drain parser output from LogHub). Label normal=0, abnormal=1.
2. **Merged vocabulary:** assign HDFS events global IDs 0–32, BGL events 33–426 →
   one 427-event space. Key each event as `domain:id` so "HDFS event 5" and "BGL
   event 5" never collide.
3. **Federation construction:** 3 HDFS + 2 BGL clients; each domain contributes 1%
   of its normal sequences as the training pool (matching the reference), split
   IID or by log-normal quantity skew.
4. **Detectors:** Length (min–max range), Known-Events (set union), ECVC (histogram
   nearest-neighbour), DeepLog (2-layer 64-unit LSTM, top-9 rule). DeepLog supports
   two inputs: **scalar ID** and **learned 16-D embedding** — the controlled variable.
5. **Experiments:**
   - *H1:* mixed federation, domain-blind vs domain-aware Length, 5 seeds.
   - *H2:* single-domain vs simultaneous-mixed DeepLog, 3 seeds, both inputs.
   - *H3:* sequential HDFS→BGL, measure forgetting + model-size growth.
   - *Mechanism:* swap scalar↔embedding to isolate the cause.
   - *Mitigation:* replay buffer sweep (5%/10%/25%/50%).
6. **Evaluation:** per-domain precision/recall/F1, forgetting, model size in bytes;
   mean ± std over seeds; reproduce single-domain baselines first (the gate).

**Tools:** Python 3.12, PyTorch (CPU), NumPy, our own FedAvg + DeepLog
implementation (we did **not** copy the reference's GPL code — clean-room reimpl).

**Why this methodology is sound (say this):** it's *controlled* — we change one
thing at a time (domain-blind vs -aware; scalar vs embedding; simultaneous vs
sequential), so every conclusion has a clean cause. And it's *honest* — a
reproduction gate up front, multi-seed everywhere, negatives reported.

### ⑤ System architecture / module design / experimental design (3 marks) — CO2, CO4

See **PART 4** for the full architecture description + the diagram prompt. In the
viva, walk them top-to-bottom through the pipeline: raw logs → parser → **merged
vocabulary (the novel bit)** → federation builder → {lightweight | deep} detectors
→ FedAvg aggregator → per-domain evaluation → results CSVs → figures/paper.
Emphasize the module boundaries in `src/`: `dataloader.py`, `vocabulary.py`,
`methods/`, `aggregation.py`, `metrics.py`.

### ⑥ Feasibility, risks, ethics & work planning (3 marks) — CO1, CO3, CO4

**Feasibility:** already demonstrated — the whole thing runs on a **laptop CPU**;
the deep model is intentionally tiny (2×64 LSTM); experiments finish in minutes to a
few hours. The experiment phase is **complete**; what remains is writing/compiling.

**Risks & how we handled them:**
| Risk | Mitigation |
|---|---|
| Can't reproduce the reference numbers | Built a reproduction *gate* first; we matched their single-domain picture before trusting mixed results. |
| A result is a lucky fluke | Multi-seed (3–5 seeds), report mean ± std. |
| Forcing a "positive" result | Explicit rule: report negatives (H2 came back negative — we kept it). |
| Fabricating numbers | Hard rule: every number traces to a `results/*.csv`. |
| Deep model doesn't hit the reference's 0.93 | Investigated honestly, wrote a Threats-to-Validity section; proved it's a capacity ceiling of the method, not our bug. |

**Ethics:** the *entire motivation* is privacy — federated learning keeps sensitive
logs on-device. We use only **public benchmark datasets** (HDFS, BGL from LogHub);
no personal data, no scraping. Our fixes don't weaken the privacy guarantee (replay
stores only the *owner's own* old data locally). We make claims we can back with
data and openly state limitations.

**Work planning / timeline (already largely executed):**
- Week 1: reproduce single-domain baselines (gate). ✅
- Week 2: build merged vocabulary + dataloader; run H1. ✅
- Week 3: DeepLog + FedAvg; H2; H3; mechanism experiment. ✅
- Week 4: ensemble scoping; multi-seed everything; two fixes (router + replay). ✅
- Now: paper writing, LaTeX compile, submission polish. ◐ (in progress)

**Responsibilities:** Vansh Mundhra, Pinak Debnath (implementation, experiments,
writing); Sankari M (guidance/review). *(Adjust to your actual split.)*

### ⑦ Preliminary report quality & response to questions (3 marks) — CO3, CO4

This is graded on your Chapters 1–3 report + how you present + how you answer.
Chapters map cleanly:
- **Chapter 1 (Introduction):** Parts 1 + 3① above (domain, problem, motivation,
  objectives, scope).
- **Chapter 2 (Literature Review):** Part 3② (the 4-bucket table + the gap).
- **Chapter 3 (Methodology / System Design):** Parts 3④ + 3⑤ + Part 4 (method,
  architecture, experimental design).
Use **PART 6** below to rehearse the Q&A.

---

## PART 4 — System architecture (in words + a text diagram)

### 4.1 The pipeline, top to bottom

```
        ┌─────────────────────┐        ┌─────────────────────┐
        │   HDFS raw logs      │        │   BGL raw logs       │
        │ (Hadoop file system) │        │ (Blue Gene/L super-  │
        │  33 event types      │        │  computer) 394 types │
        └──────────┬──────────┘        └──────────┬──────────┘
                   │  Drain parser → event-ID sequences (+ normal/abnormal label)
                   ▼                               ▼
        ┌───────────────────────────────────────────────────────┐
        │        ★ MERGED CROSS-DOMAIN VOCABULARY ★  (our novelty)│
        │  HDFS → global IDs [0,33) ,  BGL → global IDs [33,427)  │
        │  keyed as "domain:id" so IDs never collide → 427 events │
        │                    (src/dataloader.py)                  │
        └───────────────────────────┬───────────────────────────┘
                                     ▼
        ┌───────────────────────────────────────────────────────┐
        │            FEDERATION BUILDER                          │
        │  3 HDFS clients + 2 BGL clients                        │
        │  1% of normals as training pool; IID or non-IID split │
        └───────────────┬───────────────────────┬───────────────┘
                        ▼                        ▼
        ┌───────────────────────┐   ┌───────────────────────────┐
        │ LIGHTWEIGHT detectors │   │  DEEP detector: DeepLog    │
        │ • Length (min–max)    │   │  2-layer 64-unit LSTM      │
        │ • Known-Events (set)  │   │  input = scalar ID  OR     │
        │ • ECVC (histogram)    │   │          16-D embedding    │
        └───────────┬───────────┘   └─────────────┬─────────────┘
                    ▼                              ▼
        ┌───────────────────────────────────────────────────────┐
        │   AGGREGATION (FedAvg / set-union / min-max)           │
        │   ★ domain-blind  vs.  domain-AWARE (our fix) ★        │
        │                 (src/aggregation.py)                   │
        └───────────────────────────┬───────────────────────────┘
                                     ▼
        ┌───────────────────────────────────────────────────────┐
        │   PER-DOMAIN EVALUATION (precision / recall / F1,      │
        │   forgetting, model size)   (src/metrics.py)          │
        └───────────────────────────┬───────────────────────────┘
                                     ▼
                    results/*.csv  →  figures  →  IEEE Access paper
```

### 4.2 Module map (what lives where)

- `src/dataloader.py` — **the heart.** Loads both datasets, builds the merged
  vocabulary, constructs the federation. *"The reference dataloader assumes one
  dataset per run; the novel contribution starts here."*
- `src/vocabulary.py` — the merged event dictionary; `domain_of(global_id)` tells you
  which domain an event belongs to (used by the inferred router).
- `src/methods/` — the detectors (`length.py`, `known_events`, `ecvc.py`,
  `deeplog_fed.py`).
- `src/aggregation.py` — FedAvg, set-union, and our **DomainAwareLength** fix.
- `src/metrics.py` — precision/recall/F1.
- `scripts/` — one runnable file per experiment (`h1_multiseed.py`,
  `run_h2_deeplog.py`, `run_h3_sequential.py`, `h1_inferred_router.py`,
  `mitigation_replay.py`, …).
- `results/` — every CSV/figure (the source of truth for all numbers).
- `docs/paper/main.tex` — the IEEE Access paper.

### 4.3 The three "experimental designs" (the panel may ask to see these as figures)

1. **H1 design:** one mixed federation → run Length two ways (global vs
   domain-aware) → compare HDFS F1. Controlled: only the aggregation changes.
2. **H2/H3 + mechanism design:** the **2×2** — {simultaneous, sequential} ×
   {scalar-ID, embedding} → run DeepLog in all four → measure forgetting. Controlled:
   only arrival-pattern and representation change.
3. **Mitigation design:** the one failing cell → add a replay buffer of size
   5/10/25/50% → measure forgetting vs buffer size.

---

## PART 5 — Copy-paste prompts for ChatGPT / DALL·E image generation

Paste these into an image model. They're written to give clean, presentation-ready
diagrams. (Tip: ask for "16:9, white background, no gibberish text, flat vector
style, professional academic figure.")

### Prompt A — System architecture diagram (your main Chapter-3 figure)
```
Create a clean, professional academic system-architecture diagram, flat vector
style, 16:9, white background, muted blue/teal/grey palette, clear sans-serif
labels. Top row: two source boxes side by side — "HDFS logs (33 event types)" and
"BGL logs (394 event types)". Arrows labelled "Drain parser" flow down from both
into a single highlighted central box titled "Merged Cross-Domain Vocabulary
(427 events): HDFS IDs 0–32, BGL IDs 33–426". Below it a box "Federation Builder:
3 HDFS + 2 BGL clients, 1% training data, IID / non-IID". This splits into two
parallel branches: left branch "Lightweight detectors: Length, Known-Events, ECVC",
right branch "Deep detector: DeepLog LSTM (scalar-ID or 16-D embedding input)". Both
branches feed a box "Aggregation (FedAvg / set-union): domain-blind vs domain-aware".
That feeds "Per-domain Evaluation: Precision, Recall, F1, Forgetting, Model size",
which flows to a final small box "Results CSV → Figures → Paper". Emphasize the
central vocabulary box with a subtle glow/border to mark it as the novel contribution.
No decorative clutter, no fake logos.
```

### Prompt B — The 2×2 "when is it safe" map (your signature figure)
```
Create a clean 2x2 matrix infographic, flat vector academic style, 16:9, white
background. X-axis label "Arrival pattern" with two columns: "Simultaneous (mixed
at once)" and "Sequential (new domain arrives later)". Y-axis label "Input
representation" with two rows: "Scalar ID (disjoint number ranges)" and "Learned
embedding (shared feature space)". Fill three cells green with a check mark and the
word "SAFE". Fill exactly ONE cell — the intersection of "Sequential" column and
"Learned embedding" row — red with a warning icon and the text "CATASTROPHIC
FORGETTING, HDFS F1 0.70 → 0.04". Title at top: "When is federated log anomaly
detection safe? Only one cell fails." Minimal, legible, professional.
```

### Prompt C — H1 collapse-and-recover concept illustration
```
Create a simple conceptual academic illustration, flat vector, white background,
16:9. Show a number line labelled "sequence length". Draw a tight bracket labelled
"HDFS normal range (short, ~29)" and a very wide bracket labelled "BGL normal range
(huge, up to hundreds)". Show a merged bracket labelled "Domain-blind global range
(too wide — HDFS anomalies now look normal!)" spanning everything, with a sad/red
face and "HDFS F1 = 0.00". Then show two separate brackets labelled "Domain-aware:
per-domain ranges" with a happy/green face and "HDFS F1 = 0.56 restored". Clear
labels, minimal, presentation quality.
```

### Prompt D — Federated learning concept (for your intro slide)
```
Create a friendly, clean conceptual diagram of federated learning, flat vector,
16:9, white background. Center: a "Global Model" box. Around it, 5 device icons
(3 labelled "HDFS client", 2 labelled "BGL client"), each with a small local
database icon and a padlock indicating "data stays private". Arrows from each device
to the center labelled "share only model updates (not raw logs)". A return arrow
labelled "averaged model (FedAvg)". Title: "Federated learning: train locally, share
models, keep logs private". Professional, minimal, no fake brand logos.
```

### Prompt E — Catastrophic forgetting + replay fix (for H3 slide)
```
Create a clean two-panel academic illustration, flat vector, 16:9, white background.
Left panel titled "Without replay": a brain/model icon learning "BGL" while an "HDFS"
memory fades to grey, with a downward red arrow "HDFS F1 0.70 → 0.04 (forgetting)".
Right panel titled "With 10% replay buffer": the same model with a small buffer icon
labelled "~558 saved HDFS sequences", both "HDFS" and "BGL" memories bright and
retained, with a green check "HDFS F1 restored to 0.70, BGL still learned".
Minimal, legible, presentation quality.
```

> **Note:** image models sometimes render text imperfectly. For the final report,
> it's safest to generate the *layout* with these prompts, then redraw the exact
> boxes/labels in draw.io / PowerPoint / Excalidraw so the text is crisp. Prompt A
> and Prompt B are the two you most need.

---

## PART 6 — Likely viva questions & crisp answers

**Q: In one sentence, what's your contribution?**
> "We showed that two published guarantees about federated log anomaly detection
> break when a federation mixes different log domains or onboards them over time, we
> explained exactly why, and we built simple fixes for both failures."

**Q: What's actually novel — isn't this just re-running someone's code?**
> "The novelty is the *setting they never tested* — a single federation with two
> different log domains, and sequential arrival. Their public repo literally has no
> mixed config. We built the merged-vocabulary dataloader that makes it possible,
> then produced a 2×2 map of when it's safe, plus two working fixes."

**Q: Why do HDFS and BGL together break the Length method?**
> "BGL sequences vary hugely in length (up to hundreds); HDFS are short and tight.
> Merging into one global [min,max] range makes the range so wide that HDFS anomalies
> fall inside it and never get flagged — HDFS F1 goes to zero. Keeping a separate
> range per domain fixes it."

**Q: Your fix needs to know which domain a log is from — is that realistic?**
> "We don't need a manual label. Because each domain owns a disjoint block of event
> IDs, we *infer* the domain from the events themselves — and it's 100% accurate on
> every test sequence. The same structure that causes the collapse makes the domain
> recoverable."

**Q: H2 is a negative result — isn't that a failure?**
> "No — a negative that *strengthens* prior work is a real finding. Simultaneous
> mixing doesn't hurt the deep model because the two vocabularies sit in disjoint
> input ranges. Importantly, we then showed this safety is *representation-dependent*,
> which is the insight nobody had before."

**Q: What's 'catastrophic forgetting' here, and why only sometimes?**
> "When BGL arrives *after* HDFS, the network learns BGL and forgets HDFS (0.70 →
> 0.04) — but *only* with the embedding input, where the two vocabularies share a
> feature space and compete for the same weights. With the scalar-ID input they stay
> separate and nothing is forgotten. A ~10% replay buffer removes the forgetting."

**Q: Why doesn't your Length fix improve the full ensemble?**
> "Honest scoping: the ensemble is a logical OR dominated by ECVC (already F1 0.96),
> so a repaired Length adds detections the ensemble already makes. Our fix matters
> for *lightweight-only* deployments — exactly the cheap regime the reference
> motivates — where HDFS Length goes from 0.00 to 0.56."

**Q: Your DeepLog only hits F1 ~0.70, but the reference says 0.93. Bug?**
> "Not a bug — we investigated it directly. We even ran a standalone *centralized*
> DeepLog on the full HDFS data and it still doesn't reach 0.93; it's a capacity
> ceiling of the scalar-input next-event model with the top-9 rule. The 0.93 is most
> plausibly their best-tuned centralized number. Crucially, our forgetting finding is
> a *relative* effect (0.70→0.04), independent of the absolute level."

**Q: Why only two datasets?**
> "Two genuinely different domains already exhibit the failure, and adding more would
> only widen the predicted gap. More domains need new parsers — that's scoped as
> future work, not padding."

**Q: How do we know you didn't cherry-pick or fabricate numbers?**
> "Two safeguards: every number traces to a CSV in `results/`, and everything is
> multi-seeded (3–5 seeds) reported as mean ± standard deviation. We also had a
> reproduction gate — we matched the reference's single-domain numbers before
> trusting any mixed-domain result."

**Q: What are the limitations?**
> "Two domains only; DeepLog's absolute F1 is below the reference's tuned number;
> Known-Events has high seed variance on HDFS; ECVC uses a transductive threshold. We
> state all of these openly in a Threats-to-Validity section."

**Q: Where's the ethics angle?**
> "The whole point is privacy — federated learning keeps sensitive logs on-device.
> We use only public benchmarks, add no personal data, and our replay fix stores only
> the client's own past data locally, so it doesn't weaken the privacy guarantee."

---

## PART 7 — The numbers you should have memorized (all real, all in `results/`)

| Finding | Number | File |
|---|---|---|
| HDFS Length collapses under mixing | 0.538 → **0.000 ± 0.000** | `h1_multiseed.csv` |
| Domain-aware fix restores HDFS | **0.561 ± 0.024** | `h1_multiseed.csv` |
| Inferred router accuracy (no label needed) | **1.000 ± 0.000** | `h1_inferred_router.csv` |
| H2: simultaneous mixing HDFS drop (≈none) | +0.0002 ± 0.0001 | `h2_multiseed.csv` |
| H2: BGL actually *improves* when mixed | −0.106 (better) | `h2_multiseed.csv` |
| H3: embedding forgetting (catastrophic) | HDFS 0.70 → 0.04, **+0.665 ± 0.003** | `mechanism_multiseed.csv` |
| H3: scalar-ID forgetting (none) | 0.000 | `h3_sequential.csv` |
| H3: set-union grows ×3.7 (breaks C2) | 1,688 → 6,300 bytes | `h3_sequential.csv` |
| Replay fix: 10% buffer kills forgetting | +0.665 → **+0.004** | `mitigation_replay.csv` |
| Full ensemble (fix doesn't change it) | 0.984 either way | `ensemble.csv` |

Dataset facts: **HDFS** = 33 events, 558k normal / 16.8k abnormal, length 29±6.
**BGL** = 394 events, 37.8k normal / 31.4k abnormal, length 69±**747** (the 747 is
why the range explodes). Merged vocabulary = **427**.

---

### Final tip for the presentation
Lead with the *story* (Part 0 → the missing config file → the 2×2 map), show the
architecture diagram (Prompt A) and the 2×2 figure (Prompt B), then let the numbers
in Part 7 back you up. If they push on any number, point to the file. Confidence +
honesty about limitations is exactly what parameters ⑥ and ⑦ reward.
