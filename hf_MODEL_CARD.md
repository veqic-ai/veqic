---
license: apache-2.0
base_model: unsloth/gemma-4-E4B-it
language: [ar, en]
tags: [veqic, transparency, arabic, qlora, unsloth]
---

# Veqic 0.1 (Internal Beta)

> **Open, transparent AI.** Every weight, every data point, every training decision — public.
> Full transparency as the product. Breaking AI monopolies. Protecting humans from replacement and domination.

## What this is
Veqic 0.1 is an **internal beta, not a public release**. It exists to prove the concept:
a small open model fine-tuned on fully documented, human-reviewed data.

## Full transparency card
| Item | Value |
|---|---|
| Base model | Gemma 4 E4B (Apache 2.0, Google DeepMind) |
| Training method | QLoRA (r=16, alpha=16, all linear modules) via Unsloth, fp16, 3 epochs, lr 2e-4 |
| Compute | Kaggle free tier (T4 x2) — zero budget |
| Training data | 100 human-reviewed examples: 90 SFT, 5 DPO pairs, 5 eval cases — published openly |
| Dataset | [Veqic/veqic-0.1-seed](https://huggingface.co/datasets/Veqic/veqic-0.1-seed) |
| Code & methodology | [github.com/veqic-ai/veqic](https://github.com/veqic-ai/veqic) |
| Known weaknesses | Tiny data scale (100 examples) — prototype only. Not competitive with large models. Modern Standard Arabic output, dialect-aware input. |

## Personality (what we train for)
Direct and honest, no flattery. Says "I don't know" clearly. Refuses unethical requests
with explanation. Treats the user as an intellectual partner, not a customer.

## Quality gate (mandatory before any commercial launch of 1.0)
- ArabicMMLU ≥ 55% (base: 24%) • Human preference ≥ 60% on 30 real Arabic conversations • 20/20 refusal checklist
- If the gate fails: another data cycle. No weak launch.

## Roadmap
0.1 (internal, 4–8B, free compute) → 1.0 (commercial, 27B+, cheap inference) → 2.0+ (trillion-scale MoE, small active params, lowest cost).
Data is the asset that transfers between every release.
