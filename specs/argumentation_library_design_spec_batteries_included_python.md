# Goal
Build a **general-purpose, batteries-included Python library** for creating, importing, analyzing, and reasoning over **argument graphs** derived from text and multimodal evidence—without IntelliProof’s domain-specific UI assumptions.

Design targets:
- **Usable in any pipeline** (NLP, VLM, OSINT tooling, writing support, education).
- **Core graph + formal argumentation semantics** (Dung AFs, ABA) + pragmatic scoring.
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
   - **Abstract Argumentation Framework (AAF)** (Dung): extensions (grounded, preferred, stable), admissibility, labeling.
   - **Assumption-Based Argumentation (ABA)**: assumptions, rules, contraries; compute extensions, dispute trees.
   - Optional: **weighted / probabilistic / bipolar** reasoning layers.
3. **Graph algorithms**: centrality, SCCs, cycles, reachability, min-cut, clustering, temporal slicing.
4. **Ingestion / mining**:
   - Text → ADUs + relations (+ claim type) + alignment.
   - Multimodal evidence → extracted snippets/frames/regions linked to nodes.
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

### 6) `Theory` objects
- `DungAF`: (Arguments, Attacks) view derived from `ArgumentGraph`.
- `BipolarAF`: supports + attacks.
- `ABAFramework`: (Language, Rules, Assumptions, Contraries).

Key design decision: **one canonical graph object** with multiple *views*.

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
- `ArgumentGraph.to_argument_graph() -> DungAF` where nodes are bundles and edges are aggregated support/attack between bundles.
This keeps claim-level edges consistent with Dung/ABA while allowing users to reason at a higher argument abstraction layer.

---

# Library Architecture
## Layer 0 — Foundations
- `core/` dataclasses + validation
- `io/` import/export
- `viz/` renderers
- `algorithms/` graph theory

## Layer 1 — Argumentation Semantics
### AAF (Dung)
- Build `DungAF` from `ArgumentGraph` via a projection:
  - Nodes become arguments
  - `attack` edges become attacks
  - optional: map undercut/rebut to attack variants
  - when using `ArgumentBundle`, each bundle becomes an argument node; edges are aggregated across bundle boundaries

Functions:
- `extensions(semi=...)` for grounded / preferred / stable / complete
- `labelings(...)` (in/out/undec)
- `admissible_sets()`
- `skeptical_acceptance(arg)` / `credulous_acceptance(arg)`

### ABA
Provide an ABA DSL:
```python
aba = ABAFramework()
aba.add_assumption("a")
aba.add_contrary("a", "not_a")
aba.add_rule(head="p", body=["a", "q"])

result = aba.compute(semantics="preferred")
```

Under the hood:
- Translate to AF when convenient; otherwise compute directly (dispute trees).

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

---

# Public API: Minimum Delightful Set
## Construct graphs manually
```python
from arglib import ArgumentGraph

G = ArgumentGraph.new(title="Example")

c1 = G.add_claim("Green spaces reduce urban heat.", type="fact")
c2 = G.add_claim("Cities should fund parks.", type="policy")

G.add_support(c1, c2, weight=0.7, rationale="Cooling improves health")
G.attach_evidence(c1, doc_id="paper.pdf", page=3, quote="...", stance="supports")
```

## Convert to formal frameworks
```python
af = G.to_dung()
ext = af.extensions("grounded")
accepted = af.skeptically_accepted()

baf = G.to_bipolar()
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
  tasks=["grounded_extension","preferred_extensions","skeptical_acceptance"],
  explain=True
)
```

### Credibility propagation (IntelliProof-style)
- Initialize node scores from evidence values.
- Iteratively update: `c_i^(t+1) = tanh(lambda * E_i + sum(w_ji * c_j^(t)))`.
- Support edges add influence, attack edges subtract influence (using absolute source score).

## Explanations
- For AAF: return *witnesses* (defense chains, attackers defeated).
- For ABA: return dispute trees / derivations.
- For weighted layers: return score contributions (evidence + edge influence).

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
  semantics/
    dung.py
    bipolar.py
    aba/
      framework.py
      solvers.py
      dispute_trees.py
  reasoning/
    reasoner.py
    explain.py
    metrics.py
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
    test_dung.py
    test_aba.py
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

# Versioned Milestones
## v0.1 (Core)
- `ArgumentGraph` + IO JSON + Graphviz render
- Dung AF conversion + grounded/preferred/stable
- Basic diagnostics (cycles, components)
 - CLI validation + docs scaffolding

## v0.2 (ABA)
- ABA framework + at least one solver path (AF translation or direct)
- Explanations (defense chains / dispute trees)
- Edge validation + evidence scoring + credibility propagation
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
- Formal semantics primer (Dung/ABA) + practical examples
- Reproducible benchmarks and evaluation on multiple datasets

---

# Open Questions (deferred decisions)
- Canonical schema: your JSON vs JSON-LD vs AIF as default
- How to represent **argument schemes** (Walton-style) vs pure support/attack
- Weighted semantics: keep separate from formal AAF/ABA or integrate
- Model distribution: bundled weights vs optional download

