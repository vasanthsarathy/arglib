# arglib
ArgLib: Python Library for reasoning over argument graphs

## Quickstart
Install dependencies with `uv`, then run tests:
```bash
uv sync
uv run pytest
```

## CLI examples
Export a graph JSON file to DOT:
```bash
uv run arglib dot path/to/graph.json
```

Print diagnostics:
```bash
uv run arglib diagnostics path/to/graph.json
```

## Reasoner example
```python
from arglib.core import ArgumentGraph
from arglib.reasoning import Reasoner

graph = ArgumentGraph.new()
a = graph.add_claim("A")
b = graph.add_claim("B")
graph.add_attack(a, b)

reasoner = Reasoner(graph)
results = reasoner.run(["grounded_extension", "grounded_labeling"])
```
