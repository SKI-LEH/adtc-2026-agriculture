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


def main():
    base = CFG["base_model"]
    q = CFG["quant_load"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto"
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
        bf16=t["bf16"],
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
