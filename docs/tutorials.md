# Tutorials

## Build a small argument map
```python
from arglib.core import ArgumentGraph

graph = ArgumentGraph.new()
c1 = graph.add_claim("Exercise improves mood.")
c2 = graph.add_claim("People should exercise daily.")
graph.add_support(c1, c2, rationale="Mood improvement benefits health.")
```

## Analyze structure
```python
diagnostics = graph.diagnostics()
print(diagnostics["cycle_count"])
print(diagnostics["isolated_units"])
```

## Save and reload
```python
from arglib.io import save, load

save("graph.json", graph)
restored = load("graph.json", validate=True)
```

## Export DOT for visualization
```python
from arglib.viz import to_dot

dot = to_dot(graph)
print(dot)
```

## Define an argument bundle (argument-as-subgraph)
```python
bundle = graph.define_argument(
    [c1, c2],
    bundle_id="arg-1",
    metadata={"source": "doc-1"},
)
arg_graph = graph.to_argument_graph()
```

## Attach evidence cards and supporting documents
```python
from arglib.core import EvidenceCard, SupportingDocument

document = SupportingDocument(
    id="doc-1",
    name="Policy Report",
    type="pdf",
    url="https://example.com/policy.pdf",
)
graph.add_supporting_document(document)
card = EvidenceCard(
    id="ev-1",
    title="Evidence summary text.",
    supporting_doc_id=document.id,
    excerpt="Key finding...",
    confidence=0.6,
    metadata={"source_type": "report", "method": "expert"},
)
graph.add_evidence_card(card)
graph.attach_evidence_card(c1, card.id)
```

## Propagate credibility
```python
from arglib.reasoning import compute_credibility

cred = compute_credibility(graph)
print(cred.final_scores)
```

## ABA dispute trees
```python
from arglib.semantics.aba import ABAFramework

aba = ABAFramework()
aba.add_assumption("a")
aba.add_contrary("a", "not_a")
aba.add_rule("b", ["a"])
trees = aba.dispute_trees()
```

## AI evaluation helpers
```python
from arglib.ai import score_evidence, validate_edges

scores = score_evidence(graph)
edge_report = validate_edges(graph)
```

## Long-document mining (split + reconcile)
```python
from arglib.ai import ParagraphSplitter, SimpleGraphReconciler

splitter = ParagraphSplitter()
segments = splitter.split(long_text)

# Mine each segment into a graph (placeholder for a real miner).
graphs = []
for segment in segments:
    g = ArgumentGraph.new()
    claim_id = g.add_claim(segment.text)
    graphs.append(g)

reconciler = SimpleGraphReconciler()
merged = reconciler.reconcile(segments, graphs)
graph = merged.graph
```
