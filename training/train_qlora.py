#!/usr/bin/env python
"""QLoRA fine-tune of Qwen2.5-0.5B-Instruct on the agriculture chat splits.

Runs on the GPU box (Udutech credits), NOT the dev laptop. Reads config.json and
training/data/{train,eval}.jsonl (produced by prepare_data.py). Loads the base model
in 4-bit NF4, attaches a LoRA adapter, and SFT-trains on the chat-formatted turns
using TRL's SFTTrainer (which applies the Qwen2.5 chat template and masks the prompt
so loss is computed on the assistant answer only).

Deps (install on GPU box, see requirements.txt):
    torch transformers peft trl bitsandbytes accelerate datasets

Usage:
    python prepare_data.py        # once, to build data/train.jsonl + eval.jsonl
    python train_qlora.py         # writes output/adapter/
"""
import json, os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))

# bf16 needs Ampere+ (compute capability >= 8.0). On older GPUs — e.g. Colab's free
# T4 (Turing) — bf16 is unsupported, so fall back to fp16 automatically instead of
# crashing. Respects the config's intent but never picks a dtype the card can't run.
_want_bf16 = bool(CFG["train"].get("bf16")) and CFG["quant_load"]["bnb_4bit_compute_dtype"] == "bfloat16"
USE_BF16 = _want_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


def main():
    print(f"compute dtype: {'bf16' if USE_BF16 else 'fp16'} "
          f"(gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    base = CFG["base_model"]
    q = CFG["quant_load"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, torch_dtype=COMPUTE_DTYPE, device_map="auto"
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=CFG["lora"]["r"],
        lora_alpha=CFG["lora"]["alpha"],
        lora_dropout=CFG["lora"]["dropout"],
        target_modules=CFG["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    data_dir = os.path.join(HERE, CFG["data"]["out_dir"])
    ds = load_dataset(
        "json",
        data_files={"train": os.path.join(data_dir, "train.jsonl"),
                    "eval": os.path.join(data_dir, "eval.jsonl")},
    )

    t = CFG["train"]
    sft = SFTConfig(
        output_dir=os.path.join(HERE, CFG["output"]["adapter_dir"]),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"],
        lr_scheduler_type=t["lr_scheduler"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_length=t["max_seq_len"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        eval_strategy=t["eval_strategy"],
        bf16=USE_BF16,
        fp16=not USE_BF16,
        seed=t["seed"],
        report_to="none",
        packing=False,
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        peft_config=lora,
        processing_class=tok,
    )
    trainer.train()
    print(trainer.evaluate())

    out = os.path.join(HERE, CFG["output"]["adapter_dir"])
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"adapter saved -> {out}")


if __name__ == "__main__":
    main()
