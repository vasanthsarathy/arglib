# Claim + Relation Tagger Plan

Last updated: 2026-02-06
Owner: `arglib` core

## Goal
Build a high-quality, production-ready argument tagging stack for chat workflows:
- Claim/span/type tagging
- Support/attack relation extraction
- Incremental conversation memory with contradiction/revision handling
- Async deep analysis (patterns, assumptions, warrant-gated reasoning)
- Packaged model delivery for library users

## Milestones

### 1. Evaluation Harness (Baseline First)
Status: `completed`

- [x] Add repeatable evaluator for `augmented.jsonl`
- [x] Compute claim span metrics (exact + token F1)
- [x] Compute claim type macro/micro F1
- [x] Compute relation edge F1 (support/attack breakdown)
- [x] Store baseline reports in `reports/`

### 2. Dataset Expansion for Chat-Like Workloads
Status: `completed`

- [x] Define dataset schemas for:
  - `chat_grounded.jsonl` (source-backed summary/exploration)
  - `chat_exploration.jsonl` (open multi-turn reasoning)
- [x] Add scenario tags: `summarize`, `qa`, `compare`, `debate`, `research`
- [x] Build generation pipeline for chat-style examples
- [x] Add label normalization policy (`factual->fact`, rare labels->`other` as needed)
- [x] Add synthetic-only QA gates (schema validity, self-consistency, relation sanity checks)
- [x] Add synthetic perturbation suite (paraphrase/negation/reordering) for robustness scoring
- [x] Add deferred-relevance schema (`pending_items`, `resolution_links`, discourse functions)
- [x] Add synthetic deferred-relevance dataset generator
- [x] Add deferred-relevance evaluation metrics/reporting pipeline
- [x] Add grounded-evidence schema (`evidence_items`, `claim_evidence_links`)
- [x] Add synthetic grounded-evidence dataset generator for source-backed summary QA

### 3. Hybrid Tagger Upgrade
Status: `in_progress`

- [x] Keep current fast deterministic tagger as fallback
- [x] Add learned claim/span/type model (CPU-friendly)
- [x] Add learned relation model for claim pairs
- [x] Add confidence calibration + fallback routing
- [ ] Compare against baseline harness metrics
- [x] Add multitask training scaffolding (task shards, dataloaders, trainer skeleton)
- [x] Add evidence tasks to multitask shard builder (`claim_evidence_stance`, `evidence_retrieval`)
- [x] Train small-model evidence heads and integrate evidence-link inference in hybrid tagger
- [x] Add transformer training pipeline (`scripts/train_neural_tagger.py`) and neural runtime loader
- [x] Add optional neural routing in `HybridClaimRelationTagger` (`neural_model_dir`)
- [x] Validate transformer training pipeline with capped smoke run (`models/neural_tagger_smoke`)
- [x] Expand neural training coverage to remaining tasks (span/coref/deferred/discourse/NLI)
- [x] Validate added task coverage with capped smoke run (`models/neural_tagger_additional_smoke`)

### 4. Memory + Deep Async Analysis
Status: `in_progress`

- [x] Keep low-latency per-turn memory updates
- [x] Add deferred relevance lifecycle in memory (pending -> resolved with links)
- [ ] Add background deep-analysis thread/worker:
  - flaw pattern detection (`specs/argument_patterns_bank.yaml`)
  - edge validation
  - implicit assumption extraction
  - warrant-gated scoring + explanations
- [ ] Version memory updates (provisional -> revised)
- [ ] Emit UI-ready diagnostics/badges

### 5. Model Packaging + User Download Flow
Status: `in_progress`

- [x] Add model registry metadata in-repo (`arglib/data/hf_artifacts.json`)
- [ ] Add CLI for model fetch: `arglib models pull <model-id>`
- [x] Add script-based fetch flow (`scripts/download_hf_artifacts.py`)
- [ ] Add CPU default model + optional larger model path
- [x] Document first-run download + offline behavior
- [x] Add HF artifact publish flow (`scripts/publish_hf_artifacts.py`)

## Acceptance Targets
- Claim span F1 >= 0.80
- Claim type macro F1 >= 0.78
- Relation F1 >= 0.70
- Chat-turn latency: keep fast lane low-latency (target to be finalized by integration benchmark)

## Tracking Notes
- Use this file as the canonical implementation checklist.
- Update `Last updated` date and statuses every session.
- Record baseline and subsequent metric deltas in `reports/`.

## Baselines
- 2026-02-06: `reports/tagger_baseline_augmented.json` (ClaimRelationTagger v0 deterministic)

## Dataset Artifacts
- 2026-02-06: `data/arg_mining/chat_grounded.jsonl` (starter generated set)
- 2026-02-06: `data/arg_mining/chat_exploration.jsonl` (starter generated set)
- 2026-02-06: `data/arg_mining/chat_deferred_relevance.jsonl` (synthetic deferred relevance set)
- 2026-02-06: `data/arg_mining/chat_grounded_evidence.jsonl` (synthetic source evidence + claim links)
- 2026-02-06: `data/training/multitask/*.jsonl` (task-specific multitask training shards)

## Quality Reports
- 2026-02-06: `reports/chat_grounded_qc.json`
- 2026-02-06: `reports/chat_exploration_qc.json`
- 2026-02-06: `reports/chat_deferred_relevance_qc.json`
- 2026-02-06: `reports/chat_grounded_evidence_qc.json`
- 2026-02-06: `reports/tagger_robustness_chat_grounded.json`
- 2026-02-06: `reports/tagger_robustness_chat_exploration.json`
- 2026-02-06: `reports/small_tagger_training_report.json`
  - Includes evidence heads (`evidence_stance`, `evidence_retrieval`) in `models/small_tagger_v1.json`
  - Current caveat: evidence stance macro-F1 is inflated by synthetic class imbalance (mostly `support`)
- 2026-02-06: `reports/deferred_relevance_eval.json`
  - Mode: runtime-memory evaluation (ConversationMemory + HybridClaimRelationTagger)
- 2026-02-06: `reports/training/multitask_train_report.json`
- 2026-02-06: `reports/neural_tagger_smoke_report.json`
- 2026-02-06: `reports/neural_tagger_additional_smoke_report.json`

## Model Artifacts
- 2026-02-06: `models/small_tagger_v1.json` (naive Bayes claim/relation baseline)

## Demo Artifacts
- 2026-02-06: `arglib demo-ui` command and server module at `arglib/integrations/demo_ui.py`

## Packaging Artifacts
- 2026-02-06: Hugging Face artifact resolver at `arglib/ai/artifacts.py`
- 2026-02-06: Manifest stub at `arglib/data/hf_artifacts.json`
- 2026-02-06: Publish/download scripts (`scripts/publish_hf_artifacts.py`, `scripts/download_hf_artifacts.py`)
