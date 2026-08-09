# QLoRA fine-tune — Qwen2.5-0.5B agriculture

Bakes the agriculture dataset into the weights (the **only** lever on the 50%-weight
accuracy score — see `../STRATEGY.md`). Base model **Qwen2.5-0.5B-Instruct**, chosen by
the measured baseline sweep: it leads on both S_PERF and S_EFF and nothing reaches the
TPS cap on CPU, so smaller wins. Runs on the **Udutech GPU credits**, not this dev laptop.

## Files
| File | Runs where | Purpose |
|---|---|---|
| `config.json` | — | All hyperparameters + paths. Single source of truth. |
| `prepare_data.py` | anywhere (stdlib) | 200 MCQ + 80 instruction → `data/{train,eval}.jsonl` (chat format, stratified holdout). |
| `train_qlora.py` | GPU box | 4-bit NF4 QLoRA SFT → `output/adapter/`. |
| `merge_and_export.py` | GPU box (+ llama.cpp) | adapter → merged fp16 → `output/gguf/*.Q4_K_M.gguf`. |
| `requirements-gpu.txt` | GPU box | Pinned training deps. |

## Runbook
```bash
# 0. dev laptop — build the splits and eyeball the stratification (no GPU needed)
python prepare_data.py

# --- on the Udutech GPU box ---
# 1. env
pip install -r requirements-gpu.txt
# (bring data/train.jsonl + data/eval.jsonl, or re-run prepare_data.py there)

# 2. fine-tune  (~280 examples, 3 epochs — minutes on any modern GPU, well inside 5h)
python train_qlora.py                      # -> output/adapter/

# 3. merge to fp16
python merge_and_export.py --merge          # -> output/merged/

# 4. export GGUF Q4_K_M  (needs a llama.cpp checkout for convert_hf_to_gguf.py)
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
python merge_and_export.py --gguf --llama-cpp ./llama.cpp
#   -> output/gguf/agri-qwen2.5-0.5b-Q4_K_M.gguf
```

## After export — score it (dev laptop is fine for this)
Drop the GGUF into a profiler submission and compare against the stock-0.5B baseline in
`../benchmarks/candidates/qwen2.5-0.5b/` — we want S_ACC **up** with peak RSS / TPS
essentially unchanged (a LoRA merge doesn't change param count, so `params_match` stays
true and S_PERF/S_EFF should track the baseline).

## Notes / knobs
- **Answer format:** MCQ turns are trained as `"<LETTER>. <choice text>"`. That teaches the
  model to concentrate probability on the correct option — what lm-eval's log-likelihood
  `acc`/`acc_norm` rewards. Instruction turns train free-form advisory answers for the judge
  panel. Both matter; don't drop either.
- **`assistant_only_loss=True`** masks the prompt so loss is on the answer only.
- If accuracy underwhelms: bump `lora.r` 16→32, `epochs` 3→4, or widen the dataset
  (`../data/finetune/_staging/` + `build_dataset.py`), then re-run from step 0.
- **Pending gate before submission:** native-speaker review of the model-drafted `sw`/`ha`
  items (bonus is judge-verifiable).
- **Don't** train on the CPU laptop — `train_qlora.py` needs CUDA + bitsandbytes.
