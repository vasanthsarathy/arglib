# ArgLib Desktop: Electron UI Design Specification

This canvas defines a **full‑fledged desktop UI** for creating, mining, analyzing, and interacting with **Warrant‑Gated Argument Graphs (WGAGs)** using **ArgLib** as the reasoning backend and **Electron** as the application shell.

The design emphasizes:

- cognitive clarity for complex arguments
- progressive disclosure (simple graph → deep diagnostics)
- explainability of flaws and strengths
- human–AI collaboration (LLM as assistant, not judge)

---

## 0) Target users

1. Analysts / researchers (intelligence, policy, science)
2. Writers and reviewers (essays, op‑eds, legal briefs)
3. Educators / students (critical thinking)
4. Argumentation researchers

---

## 1) High‑level app layout

A **multi‑panel workspace** with a central graph and peripheral inspectors.

```
+-------------------------------------------------------+
| Menu Bar                                              |
+-------------------------------------------------------+
| Sidebar | Graph Canvas (primary)        | Inspector    |
|         |                              | Panel        |
|         |                              |               |
|         |                              |               |
+-------------------------------------------------------+
| Console / Explanation / Suggestions Panel             |
+-------------------------------------------------------+
```

---

## 2) Core panels and their responsibilities

### 2.1 Graph Canvas (center)

**Purpose:** Visualize and directly manipulate the claim graph.

**Capabilities:**

- Nodes: claims (fact / value / policy)
- Edges: support (green) / attack (red)
- Edge thickness = effective influence (S × G)
- Optional toggle: show warrant nodes inline or as edge badges

**Interactions:**

- Click node → inspect claim
- Click edge → inspect warrants + gate
- Drag nodes, auto‑layout
- Collapse / expand subgraphs

**Overlays:**

- Flaw icons on edges/nodes (⚠, ⛔)
- Strength heatmap (blue = strong, gray = weak)

---

### 2.2 Left Sidebar: Project + Navigation

**Sections:**

#### A. Project Explorer

- Argument documents
- Versions / snapshots
- Imported texts

#### B. Views

- Graph View (default)
- Evidence View
- Assumptions View
- Flaw Map
- Strength Map

#### C. Filters

- Show only weak claims
- Show only disabled edges
- Show only attacked warrants

---

### 2.3 Right Inspector Panel (context‑sensitive)

Changes content depending on selection.

#### Claim Inspector

- Claim text
- Type (fact / value / policy)
- S(claim) with explanation
- Attached evidence list
- Incoming / outgoing relations

#### Edge Inspector

- Relation type (support / attack)
- Gate mode (AND / OR)
- Warrants list with S(w)
- Gate status (active / disabled)
- Detected flaws affecting this edge

#### Warrant Inspector

- Warrant text
- Evidence supporting/undermining it
- Attacks (undercuts)
- Downstream impact analysis

---

### 2.4 Bottom Panel: Explanations & Diagnostics

**Tabs:**

#### A. Why is this weak?

Natural‑language explanation:

> "This claim is weak because its only support depends on a generalization warrant that lacks evidence."

#### B. Detected Flaws

- List of triggered patterns
- Severity (hard vs soft)
- Affected nodes/edges

#### C. Repair Suggestions

- Add evidence here
- Add missing warrant
- Split bundled claim
- Re‑route attack

#### D. Execution Log

- Gate invalidations
- Score updates
- Pattern matches

---

## 3) Menus

### 3.1 File

- New Project
- Import Text / PDF / Markdown
- Export Graph (JSON / SVG / PDF)
- Save Snapshot

### 3.2 Edit

- Undo / Redo
- Merge / Split Claims
- Promote warrant to global
- Delete / Disable node

### 3.3 Analyze

- Run Argument Mining
- Recompute Scores
- Detect Flaws
- Sensitivity Analysis
- Strength Ranking

### 3.4 View

- Toggle Warrant Visibility
- Toggle Flaw Overlays
- Layout Options
- Dark / Light mode

### 3.5 AI Assistant

- Extract claims from selection
- Propose warrants
- Find missing assumptions
- Suggest counterarguments

---

## 4) Key workflows

### Workflow 1: Text → Graph

1. Paste or import text
2. Click "Extract Argument"
3. Review proposed claims and edges
4. Accept / edit

### Workflow 2: Weakness analysis

1. Click "Analyze"
2. Flaws appear on graph
3. Click flaw icon
4. Read explanation and suggested repair

### Workflow 3: Assumption stress test

1. Select warrant
2. View downstream impact
3. Toggle warrant off
4. Observe graph collapse

---

## 5) Visual language

- Claims: rounded rectangles
- Warrants: pills or diamond badges
- Evidence: document icons
- Support: green arrows
- Attack: red arrows
- Disabled gate: dashed edge

Minimalist, technical, neutral aesthetic, monospace fonts

---

## 6) Architecture

### Frontend

- Electron + React
- React Flow for graph

### Backend

- ArgLib (Python or Node binding)
- Local LLM or API‑based

### State

- Graph state
- Score state
- Pattern state

---

## 7) Advanced capabilities (v2)

- Compare two arguments side‑by‑side
- Track argument evolution over time
- Collaborative annotation
- Teaching mode (hide AI suggestions)

---

## 8) Design philosophy (important)

> The UI does not tell users what to believe. It shows **where belief depends on fragile assumptions**, and why.

ArgLib Desktop is a **thinking tool**, not a persuasion engine.

---

## 9) Summary

This UI turns WGAGs into an interactive, explainable, human‑centered argument analysis environment that scales from casual writing to serious analytical reasoning.

