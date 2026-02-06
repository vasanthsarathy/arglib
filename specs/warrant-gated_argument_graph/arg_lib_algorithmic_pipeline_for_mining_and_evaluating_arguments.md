# ArgLib: Algorithmic Pipeline for Mining and Evaluating Arguments

This canvas specifies the **algorithmic layer** that operationalizes the ArgLib framework: how unstructured text is converted into a warrant‑gated claim graph, and how graph algorithms automatically identify **strengths, weaknesses, flaws, and repair opportunities**.

The core design principle is strict separation of roles:

- **LLMs** generate *hypotheses* (claims, relations, warrants).
- **Graph algorithms** perform *evaluation* (validity, strength, failure modes).
- **Scores and gates** are deterministic and explainable.

---

## 1) End‑to‑end processing pipeline

### Stage 0: Input
- Unstructured argumentative text (essay, memo, analysis, testimony).

---

### Stage 1: Claim and relation mining (LLM‑assisted)

**Goal:** Propose a candidate argument graph.

**Algorithms:**
- Sentence segmentation and discourse parsing.
- LLM extraction prompt:
  - Identify claims.
  - Classify claim type: fact / value / policy.
  - Identify support and attack relations.

**Output:**
- Claim nodes C.
- Directed edges R (support / attack).

> This stage is *hypothesis generation*, not evaluation.

---

### Stage 2: Warrant induction (assumption mining)

For each edge r = (u, v):

**Algorithm:**
- LLM prompt: "What assumptions must hold for u to support/attack v?"
- Generate k candidate warrants.
- Deduplicate and cluster semantically similar warrants.

**Initialization:**
- Each warrant w is a node with S(w) = 0.5 (neutral).

**Purpose:**
- Make implicit inferential structure explicit.

---

### Stage 3: Evidence grounding

**Algorithms:**
- Retrieve cited or external documents.
- Extract evidence snippets.
- Assign document trust and extraction confidence.

**Computation:**
- Evidence quality q(e) = trust(doc(e)) × conf(e).
- Evidence support Ev(x) aggregated per claim or warrant.

---

## 2) Structural semantics (gate evaluation)

### 2.1 Gate computation

For each edge r:

- gate(r) is an AND/OR formula over warrant nodes.
- Gate score G(r) := S(gate(r)).

### 2.2 Gate invalidation

Apply **formal gate invalidation policies** (see main canvas):

- DISABLE: G(r) = 0
- RESTRICT: strengthen gate conditions
- REROUTE: retarget attack
- FLAG: diagnostic only

This step enforces logical validity before scoring.

---

## 3) Epistemic strength propagation

### 3.1 Influence computation

For edge r = (u → v):

I(r) = sign(r) × S(u) × G(r)

### 3.2 Score update

Iterative update until convergence:

S(v) := σ( α · Ev(v) + β · Σ I(r) )

Where:
- σ is a logistic or tanh function.
- α and β are global weights.

This yields:
- Claim strength
- Warrant strength
- Effective edge strength

---

## 4) Automatic weakness detection algorithms

### 4.1 Structural weaknesses

Detected via graph queries:
- Cycles without independent evidence
- Unsupported conclusions
- Self‑attacks
- Redundant supports

**Effect:** gate invalidation or flags.

---

### 4.2 Warrant fragility analysis

**Algorithm:**
- For each claim v, compute sensitivity:
  - Identify warrants whose S(w) ↓ causes S(v) ↓ below threshold.

**Output:**
- "Single‑point‑of‑failure" assumptions.

---

### 4.3 Dialectical weaknesses

Detected via:
- Mis‑targeted attacks
- Authority‑only support
- Emotional or rhetorical warrants

**Effect:** attack rerouting or gate restriction.

---

## 5) Strength and robustness analysis

### 5.1 Independent support paths

Compute:
- Number of disjoint support chains
- Diversity of warrants

Strong arguments have:
- Multiple independent supports
- Non‑overlapping warrants

---

### 5.2 Defense completeness

Check whether:
- Attacks are rebutted
- Undercuts are answered

Corresponds to "fair dialectical argument" patterns.

---

## 6) Pattern‑bank execution engine

For each pattern P:

1. Match structural template.
2. Run semantic tests (LLM entailment / plausibility).
3. Apply gate action.
4. Attach explanation and repair suggestion.

This makes flaw detection **algorithmic, not judgmental**.

---

## 7) Explanations and repairs

For any weakness, ArgLib can output:

- Which gate failed
- Which warrant was responsible
- What evidence would fix it

Example:
> "This conclusion fails because the generalization warrant lacks supporting evidence. Adding a representative study would enable the edge."

---

## 8) Why this works

- LLMs propose structure.
- Graph algorithms enforce validity.
- Scores measure strength.
- Patterns explain failure modes.

No component alone decides "argument quality".

---

## 9) Algorithmic guarantees

- Convergent scoring
- Deterministic flaw handling
- Explainable outcomes
- Incremental updates

---

## 10) Summary

ArgLib operationalizes argument evaluation as:

> *Warrant‑gated graph reasoning with explicit structural semantics and quantitative evidence propagation.*

This supports fully automatic analysis of strengths, weaknesses, and fallacies in natural‑language arguments.

