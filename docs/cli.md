# CLI

ArgLib ships with a small CLI for inspecting and exporting graphs.

## Commands

### Export DOT
```bash
uv run arglib dot path/to/graph.json
uv run arglib dot path/to/graph.json --validate
```

### Diagnostics
```bash
uv run arglib diagnostics path/to/graph.json
uv run arglib diagnostics path/to/graph.json --validate
```

### Version
```bash
uv run arglib version
```

### ABA
```bash
uv run arglib aba path/to/aba.json --semantics preferred
```
