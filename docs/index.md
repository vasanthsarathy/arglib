# ArgLib Documentation

ArgLib is a Python library for creating, importing, analyzing, and reasoning over argument graphs. The library is currently in early development, but the core graph model, Dung/ABA semantics, diagnostics, and CLI are ready to use.

## What you can do today
- Build argument graphs with typed nodes and relations.
- Export graphs to JSON or Graphviz DOT.
- Run Dung-style semantics and diagnostics.
- Use a lightweight CLI for DOT export, validation, diagnostics, and ABA.
- Define argument bundles (arguments as subgraphs) and project to higher-level graphs.
- Attach evidence cards/supporting documents and propagate credibility scores.
- Run deterministic evidence scoring and edge validation helpers.
- Generate ABA dispute trees.

## Quickstart
```bash
uv sync
uv run pytest
```

## Example
```python
from arglib.core import ArgumentGraph
from arglib.reasoning import Reasoner

graph = ArgumentGraph.new(title="Parks")
c1 = graph.add_claim("Green spaces reduce urban heat.", type="fact")
c2 = graph.add_claim("Cities should fund parks.", type="policy")
graph.add_support(c1, c2, rationale="Cooling improves health")

reasoner = Reasoner(graph)
results = reasoner.run(["grounded_extension", "grounded_labeling"])
```
