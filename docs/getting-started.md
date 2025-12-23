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
