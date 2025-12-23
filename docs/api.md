# API Reference

This is a lightweight overview of the public API. Full API documentation will expand as modules stabilize.

## Core
- `ArgumentGraph`: graph container and helpers (`arglib/core/graph.py`)
- `ArgumentUnit`, `Relation`, `TextSpan`, `EvidenceItem`: core data models

## Semantics
- `DungAF`: Dung-style abstract argumentation framework (`arglib/semantics/dung.py`)
- `ABAFramework`: minimal ABA scaffolding (`arglib/semantics/aba/framework.py`)

## Reasoning
- `Reasoner`: unified reasoning entrypoint (`arglib/reasoning/reasoner.py`)

## I/O
- `dumps`/`loads`: JSON serialization (`arglib/io/json.py`)
- `validate_graph_payload`: minimal validation (`arglib/io/schema.py`)

## Visualization
- `to_dot`: Graphviz DOT export (`arglib/viz/graphviz.py`)
