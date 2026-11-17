# Veqic

Open, transparent AI. Every weight, every data point, every training decision — public.

## Mission
- Full transparency: base model, data provenance, training method, and known weaknesses published with every release.
- Breaking AI monopolies: an open alternative that no closed lab can copy.
- Protecting humans from replacement and domination by AI.

## Status
- **Veqic 0.1** — internal beta. Base: Gemma 4 E4B (Apache 2.0), fine-tuned on manually-reviewed data. Not a public release.
- Training runs entirely on free cloud compute (Kaggle T4 x2, QLoRA via Unsloth).

## Repository structure
- `docs/` — philosophy, standards, model card
- `data/` — training data schema and reviewed seed data (JSONL)
- `training/` — Kaggle-ready training notebook and scripts

## Data
All training data is human-reviewed and published here. See `data/SCHEMA.md`.
