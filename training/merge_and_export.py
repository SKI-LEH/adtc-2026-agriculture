#!/usr/bin/env python
"""Merge the LoRA adapter into fp16 weights, then export a Q4_K_M GGUF.

Step 1 (this needs torch/transformers/peft — run on the GPU box):
    python merge_and_export.py --merge
  -> writes output/merged/  (full fp16 HF model = base + adapter)

Step 2 (GGUF convert + quantize — needs a llama.cpp checkout for the converter):
    python merge_and_export.py --gguf --llama-cpp /path/to/llama.cpp
  -> runs convert_hf_to_gguf.py to fp16 GGUF, then llama-quantize to Q4_K_M.
     On Windows the prebuilt llama-quantize.exe lives at ../../tools/llama/.

You can also do both: python merge_and_export.py --merge --gguf --llama-cpp <dir>
"""
import argparse, json, os, subprocess, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
MERGED = os.path.join(HERE, CFG["output"]["merged_dir"])
GGUF_DIR = os.path.join(HERE, CFG["gguf"]["out_dir"])


def do_merge():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = CFG["base_model"]
    adapter = os.path.join(HERE, CFG["output"]["adapter_dir"])
    print(f"loading base {base} + adapter {adapter}")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    os.makedirs(MERGED, exist_ok=True)
    model.save_pretrained(MERGED, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(MERGED)
    print(f"merged fp16 model -> {MERGED}")


def find_quantize():
    win = os.path.normpath(os.path.join(HERE, "..", "..", "tools", "llama", "llama-quantize.exe"))
    if os.path.exists(win):
        return win
    found = shutil.which("llama-quantize")
    if found:
        return found
    sys.exit("llama-quantize not found (looked in ../../tools/llama and PATH)")


def do_gguf(llama_cpp):
    if not llama_cpp:
        sys.exit("--gguf requires --llama-cpp <path to llama.cpp checkout for convert_hf_to_gguf.py>")
    conv = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    if not os.path.exists(conv):
        sys.exit(f"converter not found: {conv}")
    os.makedirs(GGUF_DIR, exist_ok=True)
    fp16 = os.path.join(GGUF_DIR, "agri-qwen2.5-0.5b-f16.gguf")
    print(f"convert {MERGED} -> {fp16}")
    subprocess.check_call([sys.executable, conv, MERGED, "--outfile", fp16, "--outtype", "f16"])

    quant = CFG["gguf"]["quant"]
    out = os.path.join(GGUF_DIR, CFG["gguf"]["out_name"])
    qbin = find_quantize()
    print(f"quantize {quant} -> {out}")
    subprocess.check_call([qbin, fp16, out, quant])
    print(f"\nFINAL GGUF -> {out}")
    print("next: score it with adtc-profiler (see training/README.md step 5).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--gguf", action="store_true")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP_DIR"))
    a = ap.parse_args()
    if not (a.merge or a.gguf):
        ap.error("pass --merge and/or --gguf")
    if a.merge:
        do_merge()
    if a.gguf:
        do_gguf(a.llama_cpp)
