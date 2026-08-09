#!/usr/bin/env python
"""Convert the built MCQ + instruction sets into chat-formatted train/eval splits.

Reads paths from config.json. Emits training/data/{train,eval}.jsonl in the
messages format ({"messages":[{role,content}...]}) that train.py feeds through the
Qwen2.5 chat template. Stdlib-only and fully deterministic (seeded), so the exact
same split is reproducible on the GPU box.

MCQ -> a lettered multiple-choice turn; target is "<LETTER>. <correct choice>". This
trains the model to put probability mass on the correct option, which is what the
lm-eval log-likelihood (acc/acc_norm) scoring rewards.
Instruction -> user = instruction (+ input if present), assistant = output.

Eval holdout is stratified across (format, lang) so en/sw/ha and both formats are
represented, and is EXCLUDED from train.
"""
import json, os, random, collections

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
LETTERS = ["A", "B", "C", "D"]

def readjsonl(rel):
    p = os.path.normpath(os.path.join(HERE, rel))
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def mcq_to_chat(o):
    lines = [o["question"], ""]
    for i, c in enumerate(o["choices"]):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("")
    lines.append("Answer with the single correct option.")
    user = "\n".join(lines)
    a = o["answer"]
    assistant = f"{LETTERS[a]}. {o['choices'][a]}"
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}],
            "meta": {"fmt": "mcq", "lang": o.get("lang", "en"), "id": o["id"]}}

def ins_to_chat(o):
    user = o["instruction"]
    if o.get("input", "").strip():
        user = f"{user}\n\n{o['input'].strip()}"
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": o["output"]}],
            "meta": {"fmt": "ins", "lang": o.get("lang", "en"), "id": o["id"]}}

def main():
    rng = random.Random(CFG["data"]["seed"])
    rows = [mcq_to_chat(o) for o in readjsonl(CFG["data"]["mcq"])]
    rows += [ins_to_chat(o) for o in readjsonl(CFG["data"]["instruction"])]

    # Stratified holdout across (fmt, lang).
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[(r["meta"]["fmt"], r["meta"]["lang"])].append(r)
    holdout = CFG["data"]["eval_holdout"]
    total = len(rows)
    eval_rows, train_rows = [], []
    for key, items in buckets.items():
        rng.shuffle(items)
        n_eval = max(1, round(holdout * len(items) / total))
        n_eval = min(n_eval, len(items) - 1)  # never empty a bucket from train
        eval_rows += items[:n_eval]
        train_rows += items[n_eval:]
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    out_dir = os.path.join(HERE, CFG["data"]["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    for name, data in (("train", train_rows), ("eval", eval_rows)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

    def dist(data):
        return dict(sorted(collections.Counter((r["meta"]["fmt"], r["meta"]["lang"]) for r in data).items()))
    print(f"train: {len(train_rows)}  {dist(train_rows)}")
    print(f"eval:  {len(eval_rows)}  {dist(eval_rows)}")
    print(f"wrote -> {out_dir}\\train.jsonl , eval.jsonl")

if __name__ == "__main__":
    main()
