# Getting Started

## Prerequisites
- Python 3.11+
- `uv` installed for environment management

## Install dependencies
```bash
uv sync
```

## Run checks before pushing
```bash
scripts/check.sh
```

## Create a graph
```python
from arglib.core import ArgumentGraph

graph = ArgumentGraph.new(title="Example")
a = graph.add_claim("A")
b = graph.add_claim("B")
graph.add_attack(a, b)
```

## Serialize to JSON
```python
from arglib.io import dumps

payload = dumps(graph)
```

## Run semantics
```python
from arglib.reasoning import Reasoner

reasoner = Reasoner(graph)
results = reasoner.run(["grounded_extension"])
```

## Export DOT
```python
from arglib.viz import to_dot

dot = to_dot(graph)
```

## Evidence cards and scoring
```python
from arglib.core import EvidenceCard, SupportingDocument
from arglib.ai import score_evidence

document = SupportingDocument(
    id="doc-1",
    name="Health Report",
    type="pdf",
    url="https://example.com/report.pdf",
)
graph.add_supporting_document(document)
card = EvidenceCard(
    id="ev-1",
    title="Cooling reduces heat mortality.",
    supporting_doc_id=document.id,
    excerpt="Heat mortality falls when urban heat is reduced.",
    confidence=0.7,
    metadata={"source_type": "report", "method": "observational"},
)
graph.add_evidence_card(card)
graph.attach_evidence_card(c1, card.id)
scores = score_evidence(graph)
```

## Argument bundles
```python
bundle = graph.define_argument([c1, c2], bundle_id="arg-1")
arg_graph = graph.to_argument_graph()
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
