# ArgLib formalization: claims, evidence, warrants, assumptions, and flaws

This canvas proposes a clean formal model for your **claim-node** argument graphs with **support/attack** edges, **evidence-backed credibility**, and **implicit necessary assumptions** (warrants) that *gate* whether an edge actually “fires.” It also shows how to integrate a **flaw/pattern bank** (e.g., circularity, false cause, straw man) as *graph patterns with semantic tests*.

---

## 0) Goal

You want one model that:

- Keeps your **UI graph** simple: nodes = claims; edges = support/attack.
- Makes edges *semantically meaningful* via **implicit assumptions**.
- Handles **many-to-many** support/attack between nodes.
- Supports LLM-generated **k assumptions** per edge or per node.
- Lets “bad assumptions” **crush** an otherwise strong-looking argument.
- Avoids a proliferation of numeric scores.

---

## 1) Core objects and data model

### 1.1 Claim graph (UI-level)

Let
- \(C\) be the set of claims.
- Each claim \(c \in C\) has a type: \(\tau(c) \in \{\text{Fact},\text{Value},\text{Policy}\}\).
- Let \(R\subseteq C\times C\times\{+,-\}\) be directed relations where
  - \((u,v,+)\) means **support** from \(u\) to \(v\)
  - \((u,v,-)\) means **attack** from \(u\) to \(v\)

This is your displayed argument graph.

### 1.2 Evidence + sources

- Let \(D\) be source documents.
- Each document \(d\in D\) has a **trust score** \(T(d) \in [0,1]\) (source reliability).
- Evidence items \(e\in E\) are extracted snippets/claims from documents:
  - \(doc(e)\in D\)
  - \(conf(e)\in [0,1]\) (extraction correctness / relevance / model confidence)
  - Optional: stance \(sgn(e)\in \{+,-\}\) relative to a claim (supports / undermines)

Evidence attaches to **claims** (and, importantly, may attach to assumptions/warrants too—see below).

---

## 2) The missing semantics: warrants as “edge assumptions”

### 2.1 Idea

Each support/attack edge \((u,v,s)\) is *not unconditional*. It is valid only if certain **warrants / assumptions** hold.

Model this by introducing **warrant nodes** (often implicit) that gate the edge.

### 2.2 Warrant set per edge

For each relation \(r=(u,v,s)\in R\), let the LLM propose \(k\) candidate warrants:

- \(W_r = \{w_{r,1},\dots,w_{r,k}\}\)

Each warrant \(w\) is itself a proposition (claim-like object):
- text: “This evidence generalizes to the target population.”
- type: often Fact-ish, but can be Value/Policy warrants too.
- attachable evidence: warrants can have evidence items (citations, methodological notes, etc.).

### 2.3 How do multiple warrants combine?

You have two sensible modes; make it explicit per edge:

- **Necessary (AND) gating**: edge fires only if *all* required warrants hold.
  - \(gate(r) = \bigwedge_{w\in Req(r)} w\)
- **Sufficient set (OR) gating**: edge fires if *any one* of several warrants holds.
  - \(gate(r) = \bigvee_{w\in Alt(r)} w\)

In practice, start with **OR** among LLM proposals, then let users mark some as **required**.

### 2.4 Where do attacks go?

This is crucial: once warrants exist, you get two distinct failure modes.

- **Rebuttal**: attack the *target claim* \(v\).
- **Undercut**: attack the *warrant* \(w\), which disables the edge without directly denying \(v\).

Undercuts are exactly how “false assumptions crush an argument.”

---

## 3) What existing frameworks help?

### 3.1 Why plain Dung / bipolar is not enough by itself

Dung (attack-only) and basic Bipolar Argumentation (support+attack) are great for acceptability of nodes, but they do not naturally encode:

- “This support edge is valid only if \(w\) holds.”

You *can* emulate this by adding nodes and edges, but you end up reinventing a conditional framework.

### 3.2 Best-fit semantics layer: ADF-style acceptance conditions

**Abstract Dialectical Frameworks (ADF)** are designed for exactly this: each node has an **acceptance condition** that can mention its parents.

You can treat each claim \(v\) as having an acceptance condition that looks like:

- \(v\) is acceptable if it has at least one *active* support chain and is not defeated by sufficiently strong attacks.

In symbolic form (qualitative):

\[
acc(v) := \Big(\bigvee_{(u,v,+)\in R} (u \wedge gate(u,v,+))\Big) \wedge \neg\Big(\bigvee_{(a,v,-)\in R} (a \wedge gate(a,v,-))\Big)
\]

This encodes **warrant-gated edges** without changing your UI.

### 3.3 ABA-style assumption focus (relevant and complementary)

**Assumption-Based Argumentation (ABA)** is relevant because it elevates your warrants to first-class *assumptions*.

- Extensions correspond to coherent **assumption sets**.
- Attacks typically derive contraries of assumptions.

ABA is especially useful if you want:
- “Which background commitments does this viewpoint require?”
- “Which assumption, if rejected, collapses the conclusion?”

**Recommendation:** Use an ADF-like acceptance-condition view for computation, and an ABA-like view for *explanations* (“this conclusion relies on these assumptions”).

---

## 4) Simplifying the numeric story (strongly recommended)

Right now you have too many overlapping numbers:
- claim credibility
- evidence confidence
- document trust
- edge validity/strength
- inferential validity

This becomes hard to explain and hard to debug.

### 4.1 Replace everything with *one* core score: SUPPORT

Define a single scalar for each proposition-like object \(x\):

- \(S(x) \in [0,1]\): **support for accepting x**

Where \(x\) can be:
- a claim node \(c\)
- a warrant node \(w\)

Everything else becomes an *input* or *multiplier* used to compute \(S(\cdot)\), but not a separately displayed “score taxonomy.”

### 4.2 Evidence aggregation becomes a single “evidence support” term

For any proposition \(x\), let attached evidence be \(E_x\).

Define each evidence item’s contribution as:

- \(q(e) := T(doc(e)) \cdot conf(e)\)  (one number)

Then aggregate evidence into:

- \(Ev(x) := Agg(\{\pm q(e) : e\in E_x\})\)

Where the sign depends on whether the evidence supports or undermines \(x\).

Choose a simple aggregator:
- weighted average (clipped to [-1,1])
- or sum then \(\tanh\)

### 4.3 Edge influence becomes one thing: a gated transmission

For relation \(r=(u,v,s)\):

- gate score: \(G(r) := S(gate(r))\) where gate(r) is the AND/OR formula over warrants.
- transmitted influence: \(I(r) := sgn(s) \cdot S(u) \cdot G(r)\)
  - where \(sgn(+) = +1\), \(sgn(-) = -1\)

Now you never need separate “edge validity” and “inferential validity.”

### 4.4 Claim score update rule

Define a simple update:

\[
S(v) := \sigma\Big( \alpha\,Ev(v) + \beta\sum_{r\in In(v)} I(r) \Big)
\]

- \(\sigma\) can be logistic or rescaled tanh.
- \(\alpha,\beta\) are just 2 knobs.

That’s it.

**Interpretation:**
- Evidence and graph both contribute to a single acceptance-support score.
- Warrants affect only whether edges transmit.

### 4.5 What you display

Display only:
- **S(claim)** as “credibility”
- **S(warrant)** as “assumption strength”
- **q(e)** as “evidence quality” (optional; mostly for inspector)

This keeps the user mental model stable.

---

## 5) LLM generates k warrants: does this still work?

Yes, with two guardrails.

### 5.1 Guardrail A: treat LLM warrants as hypotheses

LLM outputs \(k\) candidate warrants \(W_r\). Do not treat them as “true.”

- Initialize each warrant with a neutral prior \(S(w)=0.5\).
- Then allow evidence and attacks to move it.

### 5.2 Guardrail B: compress/cluster warrants to avoid explosion

If every edge gets k warrants, you can quickly bloat.

Mitigations:
1) **Deduplicate** by semantic similarity (embedding clustering).
2) **Promote global warrants** (reusable across many edges):
   - e.g., “This source is credible,” “Correlation ≠ causation,” “Generalizes to this population.”
3) Let users mark warrants as:
   - global / edge-local
   - required / optional

This keeps the graph manageable and makes assumption attacks more powerful (one undercut can disable many edges).

---

## 6) Formalizing “flaws” and virtues as graph patterns (full pattern bank integration)

Your provided pattern bank spans **structural, semantic, dialectical, substructural**, and higher‑order categories (absurd, conspiratorial, dogmatic, manipulative, speculative, satirical, mythological, troll). The model above fully supports these patterns once warrants are explicit.

The key design move is this:

> **Every pattern is a test over (claims + warrants + evidence + gating), not just raw edges.**

Below is a precise mapping showing how *each class* of patterns is supported and how to operationalize them.

---

### 6.1 Unified pattern representation

Each pattern (fallacious, good_argument, absurd, conspiratorial, dogmatic, manipulative, speculative, satirical, mythological, troll) is represented as:

- **Category**: Structural | Semantic | Dialectical | Substructural | Composite
- **Graph template**: subgraph over {claim, warrant, evidence} nodes and {support, attack, undercut} relations
- **Semantic tests**: entailment, contradiction, paraphrase, causal plausibility, normativity checks (LLM‑assisted)
- **Score tests**: conditions over support scores S(·) and gate scores G(·)
- **Implication**: why this pattern weakens or strengthens an argument
- **Repair actions**: concrete suggestions (add evidence, add warrant, split claim, add rebuttal)

This schema directly accommodates your full YAML pattern bank.

---

### 6.2 Structural fallacies

| Pattern | Detection in ArgLib |
|------|-------------------|
| Circular Reasoning / Begging the Question | Support cycle where each edge has weak or generic warrants and no independent evidence |
| Self‑Attack | Claim with outgoing attack to itself |
| Unsupported Conclusion | Claim with no incoming support and low evidence support |
| Redundancy | Multiple supports from semantically equivalent claims |
| Contradiction | Mutually exclusive claims both strongly support same conclusion |

Once warrants exist, circularity can be refined: the **same warrant** appears across a support loop.

---

### 6.3 Semantic fallacies

| Pattern | Warrant‑level diagnosis |
|------|----------------|
| False Cause | Correlational support with missing/weak causal warrant |
| Non Sequitur | Edge gate score near zero due to invalid warrants |
| Hasty Generalization | Sparse instances + weak generalization warrant |
| Equivocation | Same term mapped to different senses across claims |
| Presupposition Failure | Required warrant unstated or unsupported |
| Unstated Warrant | Edge exists but no explicit warrant node |

These are primarily **gate failures**.

---

### 6.4 Dialectical fallacies

| Pattern | ArgLib interpretation |
|------|----------------------|
| Straw Man | Attack targets paraphrase not equivalent to opponent claim |
| Ad Hominem | Attack targets credibility warrant instead of claim content |
| Tu Quoque | Attack targets consistency warrant |
| Appeal to Authority | Support gated only by authority warrant |
| Bandwagon | Popularity warrant substitutes for evidence |
| Appeal to Tradition | Tradition warrant without independent justification |
| Appeal to Emotion | Emotional warrant replaces epistemic warrant |

---

### 6.5 Substructural fallacies

| Pattern | ArgLib diagnosis |
|------|----------------|
| Loaded Question | Single claim bundles hidden premises |
| Presupposition Failure | Gate depends on missing background claim |
| Unstated Warrant | Logical bridge absent from graph |
| Implicit Bias | Normative warrant unacknowledged or undefended |

Detected by comparing inferred vs explicit warrants.

---

### 6.6 Higher‑order epistemic profiles

Absurd, conspiratorial, dogmatic, manipulative, speculative, satirical, mythological, and troll patterns are detected via:

- Implausible warrants
- Self‑sealing warrant structures
- Dominance of rhetorical/emotional warrants
- Deep chains with compounding uncertainty
- Intent metadata (for satire/trolling)

These are surfaced as **epistemic risk profiles**, not strict logical errors.

---

### 6.7 Good‑argument patterns

The same machinery identifies virtues:

- Sound deduction: strong warrants, no successful undercuts
- Strong induction: many independent supports + solid generalization warrant
- Causal coherence: explicit causal warrants
- Normative justification: policy/value claims grounded in explicit ethics
- Fair dialectic: attacks answered by rebuttals or counter‑undercuts
- Transparency: no implicit warrants remain
- Concision: high support efficiency

---

### 6.8 Pattern evaluation pipeline

1. Normalize graph (materialize warrants).
2. Compute S and G scores.
3. Run structural pattern queries.
4. Run semantic/LLM tests.
5. Attach pattern annotations.
6. Surface explanations and fixes in UI.

---

### 6.9 Key insight

> Almost every pattern in your bank corresponds to a **missing, weak, misapplied, or illegitimate warrant**.

Your warrant‑gated claim graph is therefore the right semantic substrate.

---

## 6.10 Formal gate invalidation policy (per‑flaw semantics)

This section defines **exact, mechanical rules** for how detected flaws affect the graph. The guiding principle is:

> **Flaws do not subtract from scores directly. They invalidate or constrain gates, edges, or nodes.**

Formally, each flaw pattern \(P\) maps to a **gate action**:

- **DISABLE(edge)**: set \(G(r) := 0\)
- **RESTRICT(edge)**: change gate mode (e.g., OR → AND) or require stronger warrants
- **REROUTE(attack)**: retarget attack from claim to warrant (or vice‑versa)
- **FLAG(node/edge)**: annotate without semantic effect (diagnostic only)

Below is a canonical policy table aligned with your pattern bank.

---

### A. Structural flaw policies

| Flaw | Gate action | Rationale |
|---|---|---|
| Circular Reasoning | DISABLE all edges in minimal cycle unless at least one edge has independent evidence‑backed warrant | Prevents self‑support loops from transmitting strength |
| Self‑Attack | DISABLE attack edge | Self‑contradiction invalidates the relation |
| Unsupported Conclusion | FLAG node | Lack of support is epistemic, not structural |
| Redundancy | FLAG redundant edges | Does not affect validity, only efficiency |
| Contradiction | FLAG competing supports; optionally RESTRICT both edges to require stronger warrants | Contradiction requires resolution but is not automatic invalidation |

---

### B. Semantic flaw policies (warrant failures)

| Flaw | Gate action | Rationale |
|---|---|---|
| False Cause | DISABLE support edge unless causal warrant with evidence exists | Correlation cannot transmit causal support |
| Non Sequitur | DISABLE edge | No valid warrant bridges premise and conclusion |
| Hasty Generalization | RESTRICT edge: require explicit generalization warrant with sufficient evidence | Weak induction must be defended |
| Equivocation | DISABLE edge until term sense disambiguated | Meaning shift invalidates inference |
| Presupposition Failure | DISABLE edge | Required background claim missing |
| Unstated Warrant | DISABLE edge until warrant node added | Inference undefined without bridge |

---

### C. Dialectical flaw policies

| Flaw | Gate action | Rationale |
|---|---|---|
| Straw Man | DISABLE attack edge | Attack does not target actual claim |
| Ad Hominem | REROUTE attack to credibility warrant | Speaker attack does not refute proposition |
| Tu Quoque | REROUTE attack to consistency warrant | Hypocrisy does not negate truth |
| Appeal to Authority | RESTRICT support edge: authority warrant cannot be sole gate | Authority alone insufficient |
| Bandwagon | RESTRICT support edge: popularity warrant insufficient | Consensus ≠ truth |
| Appeal to Tradition | RESTRICT support edge | Historical persistence ≠ justification |
| Appeal to Emotion | DISABLE or RESTRICT edge | Emotional force is not inferential support |

---

### D. Substructural flaw policies

| Flaw | Gate action | Rationale |
|---|---|---|
| Loaded Question | DISABLE node until implicit premises split into claims | Bundled claims hide assumptions |
| Presupposition Failure | DISABLE dependent edges | Background assumption absent |
| Implicit Bias | FLAG warrant; optionally RESTRICT edge to require explicit norm defense | Bias must be acknowledged |

---

### E. Higher‑order epistemic profile policies

| Profile | Gate action | Rationale |
|---|---|---|
| Absurd / Mythological | RESTRICT all outgoing edges to require strong evidence | Implausibility raises burden |
| Conspiratorial | RESTRICT self‑sealing warrants; FLAG global hidden‑actor warrants | Prevents unfalsifiable loops |
| Dogmatic | FLAG dominance of single warrant; RESTRICT opposing attacks | One‑sided justification |
| Manipulative / Troll | FLAG graph; no automatic gate changes | Intent classification, not logic |
| Satirical | FLAG only; no gate changes | Structural mimicry without epistemic intent |

---

### F. Hard vs soft flaws

- **Hard flaws**: DISABLE edges (logical invalidity)
- **Soft flaws**: RESTRICT or FLAG (epistemic risk)

Only **hard flaws** block transmission. Soft flaws increase scrutiny but preserve user agency.

---

### G. Interaction with scoring

- Gate invalidation sets \(G(r)=0\), preventing influence transmission.
- Scores \(S(x)\) are **never directly penalized** by flaw detection.
- Explanations always reference **which gate was invalidated and why**.

This preserves separation between **structural validity** and **epistemic strength**.

---

## 7) Recommended internal schema (minimal)



### 7.1 Node
- id
- kind: claim | warrant
- claim_type: fact | value | policy (for claim kind)
- text
- evidence_ids[]
- score S(node) in [0,1]

### 7.2 Edge
- id
- src_id, dst_id
- polarity: support | attack
- warrant_ids[]   (LLM proposes k; user can mark required vs optional)
- gate_mode: AND | OR

### 7.3 Evidence
- id
- doc_id
- span/quote/region
- conf in [0,1]
- stance: supports | undermines

### 7.4 Document
- id
- name/url
- trust in [0,1]

---

## 8) Evaluation pipeline (hybrid, but simple)

1) Initialize \(S\) for claims and warrants (0.5 neutral).
2) Compute \(Ev(x)\) for each node from evidence.
3) Iteratively update \(S\) for warrants first (they are often evidence-driven).
4) Compute gate scores \(G(r)\) for each edge.
5) Iteratively update \(S\) for claims using transmitted influences \(I(r)\).
6) Run flaw pattern bank on the explicit-warrant graph.
7) Produce diagnostics:
   - “This claim is weak because these warrants are weak.”
   - “This edge is disabled by undercut attacks on its warrants.”

Optional: compute a qualitative grounded/preferred-style accept/reject/undecided using \(S\) thresholds, *only for explanation*, not as the main score.

---

## 9) What this buys you

- A precise representation of **implicit assumptions**.
- A principled way for assumptions to **crush** an argument (by disabling gates).
- A single numeric concept (\(S\)) instead of a zoo of metrics.
- A clean home for “flaws”: **pattern queries + semantic tests** over claims/warrants.
- Scalability with LLM-generated warrants through dedup + global warrant reuse.

---

## 10) Open item: pattern bank YAML link

I wasn’t able to fetch the YAML content from the provided link due to an access constraint in this environment, so the pattern integration above is written to be compatible with a typical “pattern bank” schema: `{name, description, query/template, constraints, severity, fix}`. If you paste the YAML (or upload it), I can map it 1:1 into the exact fields you already use and propose the most direct matcher implementation.

