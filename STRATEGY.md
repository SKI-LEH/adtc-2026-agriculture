# ADTC 2026 — Winning Strategy (Agriculture)
# v3 — corrected after the baseline sweep MEASURED that nothing hits the perf cap.
# See "What changed" at bottom. v2 = after reading profiler source; v1 = pre-source.

## The thesis
The scored artifact is a SINGLE GGUF file run through llama.cpp in a memory-capped
cloud VM. There is no app, no retrieval, no custom runtime in the scoring path.
Therefore: knowledge must be BAKED INTO THE WEIGHTS (fine-tuning). And because the
speed cap turned out to be UNREACHABLE on CPU (see sweep below), smaller is strictly
better on both perf and efficiency — we pick the SMALLEST model whose fine-tuned
accuracy is competitive, not the largest that "still caps."

## How scoring ACTUALLY works (read from adtc-profiler source, not the slides)
S_TOTAL = 0.50*S_ACC + 0.30*S_PERF + 0.20*S_EFF - P_THERMAL

- Evaluation runs in `--mode audit` inside a cloud VM (`measured_on: audit_cloud_vm`),
  Docker-capped at `--memory=7.5g`. It is NOT a physical laptop.
- S_ACC: lm-eval-harness tasks scored by log-likelihood (acc_norm/acc) on the RAW GGUF,
  loaded in-process via llama-cpp-python. Default smoke task = arc_easy; real audit uses
  a HIDDEN agriculture validation subset supplied by judges. PLUS judge panel assessment
  of the model's responses to 4 prompts (2 from us in metadata.json + 2 hidden).
  -> All parametric. Retrieval/app cannot touch this number.
- S_PERF = min(TPS/15, 1.0)*100. Measured by `llama-bench -ngl 0 -p 512 -n 128` (CPU only).
  Cap at TPS=15 EXISTS but is UNREACHABLE by these models on CPU (sweep below: best was
  6.93 tok/s). So S_PERF is a LIVE, LINEAR variable — every extra tok/s is +2 points, and
  smaller/faster models win. This is the correction to v2's "size up until you cap." A
  thread-scaling probe (1 vs 2 cores, +14%) shows inference is memory-bandwidth-bound, so
  more vCPU on the audit VM will NOT lift a 1.5B to the cap. => go SMALL for perf.
- S_EFF = max(0,(7 - peak_rss_gb)/7)*100. Peak RSS sampled (psutil) during the bench run.
  Lower RAM = more points. Weight 0.20.
- P_THERMAL: -10 only if throttled or core_temp >= 85C. BUT cloud VMs don't expose thermal
  sensors -> core_temp_c_peak = null -> throttled = False -> P_THERMAL = 0 in practice.
  Thermal is effectively a NON-ISSUE for the score. Do NOT spend effort here.

## Hard constraints (DQ risks)
- Peak RSS must stay under the 7.5 GB Docker cap or the process is OOM-killed = DQ.
- llama.cpp + GGUF only. runtime MUST be "llama.cpp". No other runtime is loaded.
- 100% offline during the profiling window. download_model.sh runs first, then network off.
- Weights NOT in git; *.gguf and model/ in .gitignore; weights fetched via download_model.sh
  from a public URL (HuggingFace public repo recommended).
- parameters_estimate must be within +/-15% of the GGUF's real tensor param count
  (gguf.fraud_check). Report honestly.
- Self-reported submission.json vs audit.json must reconcile: mem +/-15%, TPS +/-25%
  (>50% = FAIL). => benchmark on a ~4 vCPU / CPU-only environment so our numbers match
  the VM. Numbers from a beefy GPU box will get flagged/failed.

## Model choice — decided by the MEASURED sweep (2026-08-09)
v2 assumed every small model caps S_PERF (so perf = constant 30) and the fight was pure
S_ACC vs S_EFF. The sweep KILLED that assumption: none cap, so S_PERF is live and rewards
small. Measured on this dev box (2-core AMD A6-9210, CPU-only, Q4_K_M) — treat absolute TPS
as a RATIO between models, not the VM's final number; peak-RSS / S_EFF is CPU-independent
and directly usable:

| Model              | TPS  | S_PERF | peak RSS | S_EFF | params_match | Notes |
|--------------------|------|--------|----------|-------|--------------|-------|
| Qwen2.5-0.5B       | 6.93 | 46.2   | 0.52 GB  | 92.5  | true         | CHOSEN base |
| Llama-3.2-1B       | 4.65 | 31.0   | 1.36 GB  | 80.6  | FALSE (flag) | params header mismatch |
| Qwen2.5-1.5B       | 2.54 | 16.9   | 1.67 GB  | 76.2  | true         | v2's old primary |

Reading: 0.5B is ~2.7x faster than 1.5B and uses ~1/3 the RAM -> it leads on BOTH the 0.30
perf and 0.20 eff terms (~+29 perf, ~+16 eff of raw component points vs 1.5B here). For
1.5B to win overall it must recover that ~0.30*29 + 0.20*16 ≈ +12 S_TOTAL gap purely from
S_ACC (0.50 weight) => it needs ~+24 points of accuracy over a fine-tuned 0.5B. On a NARROW,
heavily fine-tuned agriculture domain that is not a safe bet. Also note Llama-3.2-1B trips
the params fraud_check here (params_match=false) — avoid it as base.

DECISION: base = **Qwen2.5-0.5B-Instruct** (Apache-2.0, same multilingual family, supports
the African bonus). Fine-tune it first; only revisit 1.5B if the 0.5B's post-fine-tune
accuracy is measurably short AND the audit VM turns out fast enough to shrink the perf gap.
Lock by MEASUREMENT after fine-tuning, not by argument.

## The accuracy plan (this is 50% and it is now the whole game)
Since retrieval can't help at eval time, fine-tuning is THE lever:
1. QLoRA fine-tune on curated agriculture data, formatted BOTH ways:
   - multiple-choice Q&A (matches lm-eval acc_norm/acc log-likelihood scoring)
   - instruction / short-answer Q&A (matches judge assessment of prompt responses)
   DATASET BUILT (2026-08-09) at `data/finetune/`: 200 MCQ + 80 instruction, trilingual
   (en/sw/ha), MCQ answer key balanced 50/50/50/50, independently validated. Rebuild via
   `data/finetune/build_dataset.py` (reads only `_staging/*.jsonl`). TODO: native-speaker
   review of the model-drafted sw/ha items before submission.
2. Bake in retrievable facts as parametric knowledge: crop calendars, pest/disease ID,
   livestock care, fertilizer/input guidance, weather-based and market advisory.
3. African-language (Swahili/Hausa) agri Q&A is INCLUDED — african_alpha_claim=true,
   language_scope=["en","sw","ha"]. Judges may verify -> native-speaker review pending.
4. Fits inside the 5h Udutech GPU credits.
5. Quant sweep AFTER fine-tune: Q4_K_M vs Q5_K_M vs Q3 — pick best S_ACC that keeps
   peak RSS low (S_EFF). TPS cap is moot (unreachable), so bias toward accuracy/RAM.
   Q4_K_M is the default sweet spot.

## Free points to engineer (declarative but judge-verifiable)
- african_alpha_claim = true, backed by real African-language capability.
- cross_disciplinary_pairing.load_bearing = true, with a genuine agriculture pairing
  (e.g. agronomy/extension services), not cosmetic.
- language_scope includes at least one African language code if we make the claim.
- 2 strong test_prompts that showcase the domain (judges add 2 hidden ones; don't overfit).

## RAG / the app — DEMO ONLY, not scored
Retrieval is not in the scoring path. Build a small offline advisory app w/ optional RAG
ONLY for the 2-minute video and the "build in action" screenshots (qualitative credit +
narrative). Do NOT invest in it as an accuracy mechanism. Keep it thin.

## Build order
1. Stand up the profiler locally; reproduce a run on the SmolLM2 demo submission to learn
   the exact toolchain (needs llama.cpp `llama-bench` on PATH + Python 3.11 + the accuracy
   extras). This is our measurement harness. DONE = we can score any candidate GGUF. [DONE]
2. Baseline sweep: run stock 0.5B/1B/1.5B Q4_K_M through the profiler on a CPU box.
   Record TPS, peak RSS. Confirm which caps S_PERF. [DONE 2026-08-09 — none cap; 0.5B won.]
3. Curate the agriculture dataset (MCQ + instruction, multilingual). [DONE — 200 MCQ + 80
   instruction, en/sw/ha, at data/finetune/. Pending native-speaker review of sw/ha.]
4. QLoRA fine-tune the winner (0.5B) on GPU credits. Re-measure accuracy delta. [NEXT]
5. Quant sweep -> lock final GGUF. Verify peak RSS << 7 GB.
6. Fill template: metadata.json, download_model.sh (public HF URL), REPORT.md.
7. `adtc-profiler run --mode participant` -> submission.json. Confirm measured_on ok,
   numbers plausible, params_match=true. This is Gate 1.
8. Report + 2-min video (Wi-Fi OFF) + public repo from the ADTC template.

## Open questions to confirm from challenge assets / organizers
- Exact vCPU count and CPU class of the audit VM. Now LESS decisive (inference is
  memory-bandwidth-bound; more cores ~+14% only), but still confirms the 0.5B call.
- Whether TPS_REFERENCE stays 15.0 (it's "provisional"). If it DROPS, perf could start
  capping and re-open the size question; if it rises, small wins even harder.
- Format of the hidden agriculture validation task (MCQ vs generation) — shapes fine-tune mix.
- Whether judges run the raw GGUF or any wrapper on the 4 test_prompts (assume raw GGUF).

## What changed from v2 (and why)
- v2 said S_PERF has a HARD CAP at 15 TPS, so "size up until you're about to drop below the
  cap" -> primary = 1.5B. The MEASURED sweep (2026-08-09) showed NOTHING reaches the cap on
  CPU (best 6.93 tok/s). S_PERF is therefore a live linear term that rewards small models.
- Model switched 1.5B -> **Qwen2.5-0.5B**: it leads on both perf and eff; 1.5B would need
  ~+24 S_ACC points to overcome the gap. Llama-3.2-1B dropped as a base (params_match=false).
- African bonus moved from "IF we claim" to COMMITTED (african_alpha_claim=true, sw+ha in
  scope), with the dataset built to match.

## What changed from v1 (and why)
- v1 said "RAG buys back accuracy." FALSE: the audit loads only the GGUF via llama.cpp;
  there is no retrieval hook. Accuracy is purely parametric -> fine-tuning is the lever.
- v1 worried about thermal throttling on a refurb laptop. Scoring runs in a cloud VM with
  no thermal sensors -> P_THERMAL is ~always 0. Non-issue.
- v1 treated model size as "smaller is safer." v2 refined this to a perf-cap argument; v3
  (above) corrects that too — smaller really is better, but because the cap is unreachable,
  not because of a DQ risk.
- v1 under-weighted the free declarative points (african_alpha_claim, cross-disciplinary
  pairing) and the submission<->audit reconciliation tolerance. Both now first-class.
