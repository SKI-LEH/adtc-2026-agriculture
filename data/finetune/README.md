# Agriculture Fine-Tune Dataset

Fine-tuning is the **only** lever on the 50%-weight accuracy score — the audit loads
the raw GGUF through llama.cpp with no retrieval hook, so all knowledge must be
**parametric** (baked into the weights). See `../../STRATEGY.md` §"The accuracy plan".

## Two formats, two scoring paths

The accuracy score has two components; we train for both.

| File | Format | Scores against |
|---|---|---|
| `mcq/agri_mcq.jsonl` | multiple-choice | lm-eval-harness `acc`/`acc_norm` (log-likelihood over choices) — the hidden agriculture validation subset |
| `instruction/agri_instruct.jsonl` | instruction → answer | judge-panel assessment of responses to the 4 test prompts (2 ours + 2 hidden) |

Train on **both**: MCQ teaches the model to rank the correct option under log-likelihood;
instruction data teaches it to *generate* correct, well-formed advisory answers.

## Schemas

**MCQ** (`mcq/agri_mcq.jsonl`), one JSON object per line:
```json
{"id":"mcq_maize_0001","topic":"maize","subtopic":"spacing","question":"...","choices":["...","...","...","..."],"answer":0,"lang":"en","source":"extension_general"}
```
- `answer` is the **0-indexed** position of the correct choice in `choices`.
- Keep 4 choices, one unambiguously correct. Distractors should be plausible (common
  misconceptions), not absurd — that's what makes log-likelihood scoring discriminative.

**Instruction** (`instruction/agri_instruct.jsonl`), one JSON object per line:
```json
{"id":"ins_maize_0001","topic":"maize","instruction":"...","input":"","output":"...","lang":"en","source":"extension_general"}
```
- `input` is usually empty; use it only for follow-up/context turns.
- `output` is the target answer: concise, correct, practical, framed for a smallholder
  farmer in an African context.

## Conventions
- **Language:** `lang` uses BCP-47 codes. **DECIDED: we claim the African bonus**
  (`african_alpha_claim=true`), so the mix is trilingual — English (`en`), Swahili (`sw`),
  Hausa (`ha`) — and the submission `language_scope` must list `["en","sw","ha"]`.
  The `sw`/`ha` items are model-drafted: **flag for native-speaker review before
  submission**, since the bonus is judge-verifiable and broken language can hurt more
  than help.
- **Correctness bar:** only well-established extension guidance. No invented cultivar names,
  no over-precise figures we can't stand behind. Distractors may be wrong; the keyed answer
  must not be.
- **No overfitting to our own 2 prompts:** the 2 metadata prompts (maize spacing, fall
  armyworm) are represented, but breadth across topics matters more — 2 of the 4 scored
  prompts are hidden.

## Topic coverage (seed + target)
crops (maize, cassava, sorghum/millet, cowpea/beans, groundnut, rice), pest & disease ID
(fall armyworm, cassava mosaic, striga, maize streak, late blight, aflatoxin), soil &
fertilizer (N/P/K, urea, DAP, compost, legume N-fixation), livestock (poultry/Newcastle,
cattle/tick-borne), post-harvest storage, weather/planting-calendar advisory,
conservation agriculture.

## Status
**Built to target** (2026-08-09): 200 MCQ (en:140, sw:30, ha:30) + 80 instruction
(en:54, sw:13, ha:13). MCQ answer key balanced 50/50/50/50 across positions 0-3.
Independently validated: unique ids, 4-choice/valid-answer, no empty fields.

Regenerate at any time with `python build_dataset.py`, which reads **only** from
`_staging/*.jsonl` (the raw per-topic/per-language shards — keep them; they are the
reproducible source, not temp files) and rewrites `mcq/` and `instruction/`.

**Still TODO before QLoRA:** native-speaker review of the `sw`/`ha` items (model-drafted,
bonus is judge-verifiable).
