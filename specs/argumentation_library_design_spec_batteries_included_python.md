# Goal
Build a **general-purpose, batteries-included Python library** for creating, importing, analyzing, and reasoning over **argument graphs** derived from text and multimodal evidence—without IntelliProof’s domain-specific UI assumptions.

Design targets:
- **Usable in any pipeline** (NLP, VLM, OSINT tooling, writing support, education).
- **Warrant-gated claim graphs** with evidence-driven scoring and explainable gate behavior.
- **Bridges unstructured text ↔ structured graphs** with traceability to source spans.
- **Visualization-first optional**, but not UI-bound (export to common formats).
- **Batteries included**: pretrained/fine-tuned models, prompts, patterns banks, evaluators.

Working name (placeholder): **arglib** (choose later).

Design/workflow projects 
- maintained on github. Will need a docs page (e.g., readthedocs or mkdocs or whatever)
- Good workflow -- commit, push, deploy processes 
- will need to publish on pypi as a python package 
- Will need all the testing, linting, typechecking etc. All good coding practices 
- Proper versioning and versioning 
- Make sure to incorporate CI/CD actions for github 
- And make sure to do all the precommit stuff so tests on github don't fail. 

---

# Scope
## In-scope
1. **Graph data model**: claims/ADUs, edges (support/attack/undercut), evidence attachments, metadata, provenance.
2. **Reasoning engines**:
   - **Warrant-gated credibility propagation** with claim, warrant, and gate scores.
   - **Flaw/pattern evaluation** with gate invalidation policies.
   - Optional: **weighted / probabilistic** extensions layered on top of the warrant model.
3. **Graph algorithms**: centrality, SCCs, cycles, reachability, min-cut, clustering, temporal slicing.
4. **Ingestion / mining**:
   - Text → ADUs + relations (+ claim type) + alignment.
   - Multimodal evidence → extracted snippets/frames/regions linked to nodes.
   - Long-document mining: configurable chunking, per-chunk graphs, and graph merging.
   - Graph reconciliation: deduplicate claims, resolve cross-chunk coreference, and merge edges with provenance.
5. **Quality & critique**:
   - Missing assumptions, implicit premises, fallacy/pattern detection, circularity.
   - Coherence metrics and structural diagnostics.
   - Evidence quality evaluation and edge validation (support/attack/neutral) with confidence.
   - Credibility propagation and evidence-weighted strength scoring.
6. **Visualization & export**:
   - Render to Graphviz/SVG/HTML.
   - Export/import JSON-LD, AIF, DOT, NetworkX.
7. **Evaluation harness**:
   - Datasets adapters; model benchmarking; reproducible scoring.
   - Pattern classification and argumentation dataset statistics.

## Out-of-scope (v1)
- Full interactive web app; multi-user collaboration; storage/permissions (can be separate projects).
- Full-blown knowledge base management (provide hooks instead).

---

# Core Concepts & Data Model
## Entities
### 1) `TextSpan`
Represents a grounded piece of text (or OCR output) with provenance.
```python
TextSpan(
  doc_id: str,
  start: int,
  end: int,
  text: str,
  page: int | None = None,
  bbox: tuple[float,float,float,float] | None = None,
  modality: Literal["text","pdf","image","video","audio"] = "text",
)
```

### 2) `EvidenceItem`
Any support/attack evidence attached to a node.
```python
EvidenceItem(
  id: str,
  source: TextSpan | dict,   # dict for structured evidence (tables, EXIF, etc.)
  stance: Literal["supports","attacks","neutral"],
  strength: float | None,    # optional model-estimated
  quality: dict = {},        # e.g., reliability, timeliness, corroboration
)
```

### 2a) `SupportingDocument` (IntelliProof-style)
Represents a source document for evidence cards.
```python
SupportingDocument(
  id: str,
  name: str,
  type: str,
  url: str,
  metadata: dict | None = None,
)
```

### 2b) `EvidenceCard` (IntelliProof-style)
A structured evidence item with excerpts and confidence.
```python
EvidenceCard(
  id: str,
  title: str,
  supporting_doc_id: str,
  excerpt: str,
  confidence: float,  # [-1, 1] or [0, 1] depending on task
)
```

### 3) `ArgumentUnit` (ADU / claim)
```python
ArgumentUnit(
  id: str,
  text: str,
  type: Literal["fact","value","policy","other"] = "other",
  spans: list[TextSpan] = [],
  evidence: list[EvidenceItem] = [],
  evidence_ids: list[str] = [],  # optional link to EvidenceCard entries
  evidence_min: float | None = None,
  evidence_max: float | None = None,
  metadata: dict = {},       # speaker, time, source confidence, etc.
)
```

### 4) `Relation`
```python
Relation(
  src: str,
  dst: str,
  kind: Literal["support","attack","undercut","rebut"],
  weight: float | None = None,
  rationale: str | None = None,
  metadata: dict = {},
)
```

### 5) `ArgumentGraph`
A typed, provenance-aware multigraph.
```python
ArgumentGraph(
  units: dict[str, ArgumentUnit],
  relations: list[Relation],
  metadata: dict = {},       # title, topic, scenario, timestamps
)
```

### 6) Warrant-gated objects
- `Warrant`: assumption node that gates whether an edge transmits.
- `WarrantAttack`: undercut edge from a claim to a warrant.
- `Relation.warrant_ids` + `Relation.gate_mode` define the gate formula per edge.

Key design decision: **one canonical claim graph** with explicit warrants and gates.

### 7) `ArgumentBundle` (argument as subgraph)
In IntelliProof-style graphs, nodes are **claims** and edges are **support/attack**. In abstract argumentation, an **argument** is an atomic unit. ArgLib will support a higher-level abstraction where an **argument** can be defined as a connected subgraph (at least two claims with a relation), and a derived graph is built over these arguments.
```python
ArgumentBundle(
  id: str,
  units: list[str],          # claim ids included in the argument
  relations: list[Relation], # internal edges
  metadata: dict = {},
)
```
API direction:
- `ArgumentGraph.define_argument(units=[...], relations=[...]) -> ArgumentBundle`
- `ArgumentGraph.to_argument_graph() -> ArgumentBundleGraph` where nodes are bundles and edges are aggregated support/attack between bundles.
This keeps claim-level edges consistent while allowing users to reason at a higher argument abstraction layer.

---

# Library Architecture
## Layer 0 — Foundations
- `core/` dataclasses + validation
- `io/` import/export
- `viz/` renderers
- `algorithms/` graph theory

## Layer 1 - Warrant-gated Semantics
- Evidence support is computed for claims and warrants.
- Warrant scores gate edge transmission (AND/OR).
- Claim scores update from evidence plus gated influences.
- Flaw patterns invalidate or restrict gates (diagnostics, not direct score penalties).

## Layer 2 — AI-Assisted Mining & Critique
### Argument mining pipeline (end-to-end)
```python
miner = ArgumentMiner(model="arglib/adu-rel-qwen14b")
G = miner.parse(text, return_alignment=True)
```
Subtasks:
1. ADU extraction
2. Relation extraction
3. Relation classification
4. Claim type classification
5. Alignment (node ↔ text spans)

### Long-document mining pipeline
1. Split documents into segments (by tokens/sections/paragraphs/custom).
2. Mine each segment into a subgraph with local provenance.
3. Reconcile across segments:
   - merge duplicate or coreferent claims,
   - unify evidence cards and sources,
   - merge or aggregate edges with provenance.
4. Emit a unified graph + mapping back to source segments.
Expose `Splitter`, `GraphReconciler`, and `MergePolicy` interfaces so users can customize behavior.

### Assumption generation
```python
assumptions = AssumptionGenerator(model="arglib/assumptions-small")
assumptions.suggest(G, edge=("c1","c2"), k=3)
```

### Pattern/fallacy detection
- A YAML/JSON pattern bank + ML classifier.
- `PatternMatcher.detect(G)` returns pattern matches with subgraph bindings.

### IntelliProof-inspired analysis
- Edge validation: classify support/attack/neutral with reasoning + confidence in [-1, 1].
- Evidence checking: score evidence vs claim with reasoning + confidence.
- Claim type classification with confidence scores.
- Credibility propagation: evidence-initialized scores + weighted edge influence, iterative tanh.
- Graph critique and reporting: argument flaws, pattern matches, and summary reports.

## Layer 3 — Applications & Integrations
- `integrations/` for:
  - LangChain/LlamaIndex adapters
  - TransformerLens hooks (optional)
  - VLM evidence extractors
  - streamlit widgets (optional)

### UI + Service integration (separate repos)
- Keep `arglib` as the core library.
- Create `arglib-server` (FastAPI) to expose graph CRUD, mining, diagnostics,
  credibility, critique, and export endpoints.
- Create `arglib-ui` as a web app for graph authoring, evidence management,
  assumption review, and analysis reports.
- UI consumes server APIs; server uses `arglib` for all logic.

---

# Public API: Minimum Delightful Set
## Construct graphs manually
```python
from arglib import ArgumentGraph
from arglib.core import EvidenceCard, SupportingDocument

G = ArgumentGraph.new(title="Example")

c1 = G.add_claim("Green spaces reduce urban heat.", type="fact")
c2 = G.add_claim("Cities should fund parks.", type="policy")

G.add_support(c1, c2, weight=0.7, rationale="Cooling improves health")
doc = SupportingDocument(
  id="doc-1",
  name="paper.pdf",
  type="pdf",
  url="https://example.com/paper.pdf",
)
G.add_supporting_document(doc)
card = EvidenceCard(
  id="ev-1",
  title="Evidence excerpt",
  supporting_doc_id=doc.id,
  excerpt="Cooling improves health outcomes.",
  confidence=0.7,
)
G.add_evidence_card(card)
G.attach_evidence_card(c1, card.id)
```

## Compute warrant-gated scores
```python
from arglib.reasoning import compute_credibility

cred = compute_credibility(G)
claim_scores = cred.final_scores
warrant_scores = cred.warrant_scores
gate_scores = cred.gate_scores
```

## Run diagnostics
```python
diag = G.diagnostics()
# cycles, contradictions, disconnected components, weakly supported claims, etc.

crit = G.critique()
# missing assumptions, fallacy matches, circularity reports
```

## Visualize / export
```python
G.render("graph.svg", engine="graphviz")
G.render("graph.html", engine="pyvis")
G.save("graph.json")
G.save("graph.aif.json")
```

## Mine from text (batteries)
```python
from arglib.ai import ArgumentMiner

miner = ArgumentMiner(model="arglib/miner")
G = miner.parse(document_text)
```

---

# Reasoning Interfaces
## Unified `Reasoner`
```python
from arglib.reasoning import Reasoner

r = Reasoner(graph=G)
res = r.run(
  tasks=["credibility", "claim_scores", "warrant_scores", "gate_scores"],
  explain=True
)
```

### Credibility propagation (IntelliProof-style)
- Initialize node scores from evidence values.
- Iteratively update: `c_i^(t+1) = tanh(lambda * E_i + sum(w_ji * c_j^(t)))`.
- Support edges add influence, attack edges subtract influence (using absolute source score).

## Explanations
- Return score contributions (evidence + gated edge influence).
- Reference which warrants gated each edge and which gates were disabled.

---

# AI Models & “Batteries Included” Strategy
Provide two modes:
1. **Offline-first**: open-weight local models via Transformers/Ollama.
2. **Bring-your-own LLM**: adapters for OpenAI/Anthropic/etc.

Deliverables:
- `arglib-models` package or optional extra: `pip install arglib[models]`
- Model cards + prompts + evaluation scripts.

Suggested model set:
- `miner-small` (fast): ADU + relations, good for drafts
- `miner-accurate`: bigger, slower
- `assumption-small`: generates implicit premises
- `pattern-classifier`: pattern/fallacy category
- `evidence-extractor`: PDF/image chunking + relevance ranking

Dataset alignment:
- Provide adapters and training recipes for your **Absurd-Arguments** style graph-first generation (keeps structure stable).

### Dataset generation & augmentation (argument-mining)
- Graph-first dataset generation with pattern/fallacy templates.
- Argument graph parsing from generated essays.
- Linguistic style augmentation (formality, tone, fluency, verbosity, directness).
- Dataset statistics and pattern classification utilities.

---

# File/Package Layout (proposed)
```
arglib/
  __init__.py
  core/
    graph.py
    units.py
    relations.py
    evidence.py
    spans.py
    validators.py
  io/
    json.py
    jsonld.py
    aif.py
    dot.py
    networkx.py
  viz/
    graphviz.py
    pyvis.py
    matplotlib.py
    themes.py
  algorithms/
    basics.py
    centrality.py
    cycles.py
    flow.py
    clustering.py
  reasoning/
    reasoner.py
    warrant_gated.py
    credibility.py
  critique/
    assumptions.py
    patterns.py
    fallacies.py
    coherence.py
  ai/
    miner.py
    models.py
    prompts/
    scoring.yaml
    mining.yaml
  data/
    patterns_bank.yaml
  integrations/
    langchain.py
    llamaindex.py
    vlm.py
  cli/
    main.py
    commands.py
  tests/
    test_core.py
    test_io.py
    test_miner.py
```

---

# Design Principles
1. **Traceability is mandatory**: every claim can point back to spans/evidence.
2. **Views not forks**: one canonical `ArgumentGraph`, multiple formal projections.
3. **Determinism where possible**: reasoning engines deterministic; AI components configurable and cached.
4. **Extensibility**: plugins for new semantics, new miners, new evidence modalities.
5. **Interoperability**: AIF + JSON-LD export; works with NetworkX.

---

# Design Decisions (Locked)
These choices are fixed unless explicitly revised.

## Evidence scoring location
- Store evidence strength on both `EvidenceItem` and `EvidenceCard`.
- `EvidenceCard.confidence` is canonical; sync into `EvidenceItem.strength` on attach.

## Edge validation + weights
- `Relation.weight` is the canonical edge weight for propagation/graph algorithms.
- LLM confidence is stored in `Relation.metadata["confidence"]` with reasoning in `Relation.rationale`.

## Claim graph vs. argument graph projection
- Aggregate cross-bundle edges by summing signed weights and clamping to [-1, 1].
- Provide alternate aggregation modes (mean, max, softmax) in API.

## Claim vs. argument abstraction
- Claims are the canonical nodes in `ArgumentGraph`.
- Arguments are defined as bundles of claims (connected subgraphs).
- Bundles provide an optional higher-level abstraction for aggregation.

## Evidence score ranges
- Use [0, 1] for evidence confidence and document trust.
- Use [0, 1] for claim and warrant support scores.
- Use [-1, 1] only for signed stance (support/attack) when needed.

---

# Versioned Milestones
## v0.1 (Core)
- `ArgumentGraph` + IO JSON + Graphviz render
- Warrant-gated credibility propagation + gate scores
- Basic diagnostics (cycles, components)
- CLI validation + docs scaffolding

## v0.2 (Warrant-gated diagnostics)
- Warrant-gated explanations and gate invalidation policies
- Edge validation + evidence scoring
- Flaw/pattern detection + repair suggestions
- Argument bundling (argument = subgraph) + projection to argument graph

## v0.3 (Batteries)
- Argument miner baseline (open-weight) + caching
- Assumption generator
- Pattern bank + matcher
- Graph-first dataset generator + linguistic style augmentation
- Pattern classification and dataset stats tooling

## v0.4 (Multimodal)
- Evidence extractors for PDF/image (chunking + linking)
- Graph provenance enhancements
- Evidence card + supporting document pipelines

## v1.0
- Stable API, docs, tutorials, benchmark scripts, model cards
- Integrations (LangChain/LlamaIndex), evaluation harness, and reporting

---

# Documentation Plan
- Quickstart: build → reason → visualize
- Cookbook:
  - from essays
  - from chat transcripts
  - from OSINT doc bundles
  - from multimodal reports
- Warrant-gated reasoning primer + practical examples
- Reproducible benchmarks and evaluation on multiple datasets

---

# Open Questions (deferred decisions)
- Canonical schema: your JSON vs JSON-LD vs AIF as default
- How to represent **argument schemes** (Walton-style) vs pure support/attack
- Weighted semantics: keep separate from warrant-gated scoring or integrate
- Model distribution: bundled weights vs optional download

   - Graph reconciliation: deduplicate claims, resolve cross-chunk coreference, and merge edges with provenance.





