# ruff: noqa: E501, I001

"""Lightweight local demo UI for argument tagging and memory updates."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from arglib.ai import ConversationMemory, HybridClaimRelationTagger
from arglib.core import ArgumentGraph
from arglib.critique import detect_patterns
from arglib.reasoning import compute_credibility


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ArgLib Demo</title>
  <style>
    :root {
      --fact: #d9efff;
      --value: #ffe7c2;
      --policy: #dff5d8;
      --other: #ececec;
      --issue: #cc2f2f;
    }
    body { font-family: -apple-system, Segoe UI, sans-serif; margin: 24px; background: #f7f7f5; color: #1d1d1d; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
    .card { background: white; border: 1px solid #ddd; border-radius: 10px; padding: 14px; }
    textarea { width: 100%; min-height: 140px; font-size: 14px; }
    button { padding: 8px 12px; margin-right: 8px; border-radius: 8px; border: 1px solid #999; background: #fff; cursor: pointer; }
    pre { white-space: pre-wrap; word-wrap: break-word; font-size: 12px; background: #f0f0ee; padding: 10px; border-radius: 8px; }
    .badge { display: inline-block; background: #ececec; border-radius: 999px; padding: 2px 8px; margin-right: 6px; font-size: 12px; }
    .legend span { padding: 2px 8px; border-radius: 8px; margin-right: 6px; font-size: 12px; border: 1px solid #ccc; }
    .fact { background: var(--fact); }
    .value { background: var(--value); }
    .policy { background: var(--policy); }
    .other { background: var(--other); }
    #highlighted { line-height: 1.8; border: 1px solid #ddd; border-radius: 8px; padding: 10px; min-height: 90px; }
    .claim-tag { border-radius: 6px; padding: 1px 5px; border: 1px solid #bbb; }
    .issue { color: var(--issue); font-weight: 600; margin-left: 6px; }
    #graph { width: 100%; height: 380px; border: 1px solid #ddd; border-radius: 8px; background: #fff; }
    .claim-list { list-style: none; padding: 0; margin: 0; max-height: 280px; overflow: auto; }
    .claim-list li { border-bottom: 1px dashed #ddd; padding: 6px 0; font-size: 13px; }
  </style>
</head>
<body>
  <h2>ArgLib Claim + Relation Tagger Demo</h2>
  <div class="card">
    <textarea id="text">The model is useful. Therefore we should deploy it. However, it is not reliable in all domains.</textarea>
    <div style="margin-top:10px;">
      <button onclick="analyze()">Analyze Turn</button>
      <button onclick="resetMemory()">Reset Memory</button>
      <span class="badge" id="mode">memory:on</span>
    </div>
    <div class="legend" style="margin-top:10px;">
      <span class="fact">fact</span>
      <span class="value">value</span>
      <span class="policy">policy</span>
      <span class="other">other</span>
      <span style="border:none;">⚠ = issue</span>
    </div>
    <h4>Tagged Text</h4>
    <div id="highlighted"></div>
  </div>
  <div class="row" style="margin-top:16px;">
    <div class="card"><h3>Summary</h3><pre id="summary"></pre></div>
    <div class="card"><h3>Issues / Patterns</h3><pre id="patterns"></pre></div>
  </div>
  <div class="row" style="margin-top:16px;">
    <div class="card">
      <h3>Memory Graph</h3>
      <svg id="graph" viewBox="0 0 640 380"></svg>
    </div>
    <div class="card">
      <h3>Claims In Memory</h3>
      <ul id="claimList" class="claim-list"></ul>
      <h3 style="margin-top:12px;">Relations</h3>
      <pre id="relations"></pre>
    </div>
  </div>
  <script>
    const TYPE_CLASS = {fact: "fact", value: "value", policy: "policy", other: "other"};

    function escapeHtml(text) {
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function renderHighlighted(data) {
      document.getElementById("highlighted").innerHTML = data.highlighted_html || escapeHtml(document.getElementById("text").value);
    }

    function renderClaimList(claims, issueMap) {
      const list = document.getElementById("claimList");
      list.innerHTML = "";
      for (const claim of claims) {
        const li = document.createElement("li");
        const issues = issueMap[claim.id] || [];
        const issueIcon = issues.length ? `<span class="issue">⚠ ${issues.join(", ")}</span>` : "";
        li.innerHTML = `<span class="claim-tag ${TYPE_CLASS[claim.type] || "other"}">${claim.type}</span> <b>${claim.id}</b>: ${escapeHtml(claim.text)} ${issueIcon}`;
        list.appendChild(li);
      }
    }

    function renderGraph(claims, relations, issueMap) {
      const svg = document.getElementById("graph");
      const width = 640;
      const height = 380;
      const cx = width / 2;
      const cy = height / 2;
      const radius = Math.max(90, Math.min(width, height) / 2 - 70);
      const nodes = claims.map((c, i) => {
        const angle = (2 * Math.PI * i) / Math.max(claims.length, 1);
        return { ...c, x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
      });
      const byId = {};
      for (const n of nodes) byId[n.id] = n;

      let out = `<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#666"></path></marker></defs>`;
      for (const rel of relations) {
        const src = byId[rel.src];
        const dst = byId[rel.dst];
        if (!src || !dst) continue;
        const color = rel.kind === "attack" ? "#c0392b" : "#2e7d32";
        out += `<line x1="${src.x}" y1="${src.y}" x2="${dst.x}" y2="${dst.y}" stroke="${color}" stroke-width="2" marker-end="url(#arrow)" />`;
      }
      for (const n of nodes) {
        const cls = TYPE_CLASS[n.type] || "other";
        const fill = cls === "fact" ? "#d9efff" : cls === "value" ? "#ffe7c2" : cls === "policy" ? "#dff5d8" : "#ececec";
        const issues = issueMap[n.id] || [];
        out += `<circle cx="${n.x}" cy="${n.y}" r="22" fill="${fill}" stroke="#777" />`;
        out += `<text x="${n.x}" y="${n.y + 4}" text-anchor="middle" font-size="11">${n.id}</text>`;
        if (issues.length) {
          out += `<text x="${n.x + 18}" y="${n.y - 14}" text-anchor="middle" font-size="14" fill="#cc2f2f">⚠</text>`;
        }
      }
      svg.innerHTML = out;
    }

    async function analyze() {
      const text = document.getElementById("text").value;
      const resp = await fetch("/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text, speaker: "assistant", use_memory: true})
      });
      const data = await resp.json();
      renderHighlighted(data);
      document.getElementById("summary").textContent = JSON.stringify(data.summary, null, 2);
      document.getElementById("relations").textContent = JSON.stringify(data.relations, null, 2);
      document.getElementById("patterns").textContent = JSON.stringify(data.patterns, null, 2);
      renderClaimList(data.claims, data.claim_issue_map || {});
      renderGraph(data.claims, data.relations, data.claim_issue_map || {});
    }

    async function resetMemory() {
      await fetch("/reset", {method: "POST"});
      document.getElementById("summary").textContent = "memory reset";
      document.getElementById("highlighted").textContent = "";
      document.getElementById("relations").textContent = "";
      document.getElementById("patterns").textContent = "";
      document.getElementById("claimList").innerHTML = "";
      document.getElementById("graph").innerHTML = "";
    }
  </script>
</body>
</html>
"""


class DemoState:
    def __init__(
        self,
        model_path: str | None,
        *,
        auto_download_artifacts: bool = False,
        artifact_manifest_path: str | None = None,
        artifact_cache_dir: str | None = None,
    ) -> None:
        self.tagger = HybridClaimRelationTagger(
            model_path=model_path,
            auto_download_artifacts=auto_download_artifacts,
            artifact_manifest_path=artifact_manifest_path,
            artifact_cache_dir=artifact_cache_dir,
        )
        self.memory = ConversationMemory(tagger=self.tagger)

    def reset(self) -> None:
        self.memory = ConversationMemory(tagger=self.tagger)


def _graph_summary(graph: ArgumentGraph) -> dict[str, Any]:
    diagnostics = graph.diagnostics()
    credibility = compute_credibility(graph)
    return {
        "units": len(graph.units),
        "relations": len(graph.relations),
        "cycle_count": diagnostics["cycle_count"],
        "unsupported_claims": diagnostics["unsupported_claims"],
        "top_scores": sorted(
            credibility.final_scores.items(), key=lambda item: item[1], reverse=True
        )[:5],
    }


def _claim_issue_map(
    patterns: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
) -> dict[str, list[str]]:
    issue_map: dict[str, list[str]] = {}
    for item in contradictions:
        for key in ("new_claim_id", "prior_claim_id"):
            claim_id = str(item.get(key, ""))
            if not claim_id:
                continue
            issue_map.setdefault(claim_id, []).append("contradiction")
    for pattern in patterns:
        pattern_id = str(pattern.get("pattern_id", "pattern"))
        for claim_id in pattern.get("nodes", []):
            cid = str(claim_id)
            if not cid:
                continue
            issue_map.setdefault(cid, []).append(pattern_id)
    for cid, issues in list(issue_map.items()):
        issue_map[cid] = sorted(set(issues))
    return issue_map


def _highlighted_html(text: str, claims: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    cursor = 0
    sorted_claims = sorted(claims, key=lambda item: int(item.get("start", 0)))
    for claim in sorted_claims:
        start = int(claim.get("start", 0))
        end = int(claim.get("end", 0))
        if start < cursor or start >= end or end > len(text):
            continue
        pieces.append(_escape_html(text[cursor:start]))
        claim_text = _escape_html(text[start:end])
        claim_type = str(claim.get("type", "other"))
        issue_icon = " <span class='issue'>⚠</span>" if claim.get("has_issue") else ""
        pieces.append(
            f"<span class='claim-tag {claim_type}'>{claim_text}{issue_icon}</span>"
        )
        cursor = end
    pieces.append(_escape_html(text[cursor:]))
    return "".join(pieces).replace("\n", "<br>")


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def run_server(
    *,
    host: str,
    port: int,
    model_path: str | None,
    auto_download_artifacts: bool = False,
    artifact_manifest_path: str | None = None,
    artifact_cache_dir: str | None = None,
) -> None:
    state = DemoState(
        model_path=model_path,
        auto_download_artifacts=auto_download_artifacts,
        artifact_manifest_path=artifact_manifest_path,
        artifact_cache_dir=artifact_cache_dir,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML.encode("utf-8"))
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/reset":
                state.reset()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')
                return
            if self.path != "/analyze":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body)
            text = str(payload.get("text", "")).strip()
            speaker = str(payload.get("speaker", "assistant"))
            use_memory = bool(payload.get("use_memory", True))
            if not text:
                self._json({"error": "text is required"}, status=400)
                return

            if use_memory:
                update = state.memory.ingest_turn(text, speaker=speaker)
                graph = state.memory.graph
                turn_id = update.turn_id
            else:
                tagged = state.tagger.tag(text, doc_id="demo")
                update = None
                graph = tagged.graph
                turn_id = "demo"

            patterns = [item.to_dict() for item in detect_patterns(graph)]
            contradictions = update.contradictions if update else []
            issue_map = _claim_issue_map(patterns, contradictions)
            claims = [
                {
                    "id": unit.id,
                    "text": unit.text,
                    "type": unit.type,
                    "metadata": unit.metadata,
                }
                for unit in graph.units.values()
            ]
            turn_claims = []
            for unit in graph.units.values():
                for span in unit.spans:
                    if span.doc_id != turn_id:
                        continue
                    if span.start < 0 or span.end > len(text):
                        continue
                    turn_claims.append(
                        {
                            "id": unit.id,
                            "text": span.text or unit.text,
                            "type": unit.type,
                            "start": span.start,
                            "end": span.end,
                            "has_issue": bool(issue_map.get(unit.id)),
                        }
                    )
            relations = [
                {
                    "src": rel.src,
                    "dst": rel.dst,
                    "kind": rel.kind,
                    "weight": rel.weight,
                    "rationale": rel.rationale,
                }
                for rel in graph.relations
            ]
            response = {
                "summary": _graph_summary(graph),
                "claims": claims,
                "relations": relations,
                "highlighted_html": _highlighted_html(text, turn_claims),
                "claim_issue_map": issue_map,
                "patterns": {
                    "matches": patterns,
                    "contradictions": contradictions,
                },
            }
            if update:
                response["turn_update"] = {
                    "turn_id": update.turn_id,
                    "added_claim_ids": update.added_claim_ids,
                    "merged_claim_ids": update.merged_claim_ids,
                    "added_relations": update.added_relations,
                    "pending_created": update.pending_created,
                    "pending_resolved": update.pending_resolved,
                }
            self._json(response)

        def log_message(self, _format: str, *args: Any) -> None:
            return

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            content = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"ArgLib demo UI running on http://{host}:{port}")
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-path", default="models/small_tagger_v1.json")
    parser.add_argument("--auto-download-artifacts", action="store_true")
    parser.add_argument("--artifact-manifest-path", default=None)
    parser.add_argument("--artifact-cache-dir", default=None)
    args = parser.parse_args()
    run_server(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        auto_download_artifacts=args.auto_download_artifacts,
        artifact_manifest_path=args.artifact_manifest_path,
        artifact_cache_dir=args.artifact_cache_dir,
    )


__all__ = ["main", "run_server"]
