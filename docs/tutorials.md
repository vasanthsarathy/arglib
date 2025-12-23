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
