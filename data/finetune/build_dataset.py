#!/usr/bin/env python
"""Merge staged agri fine-tune shards into the final MCQ + instruction sets.

Reads ONLY data/finetune/_staging/*.jsonl (seed shards included there), so it is
safe to re-run. Writes the final mcq/agri_mcq.jsonl and instruction/agri_instruct.jsonl.

Pipeline: validate -> dedup (normalized text) -> MCQ answer-position rebalance
(round-robin, deterministic) -> trim to target keeping all non-English items.
"""
import json, re, glob, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
STAGING = os.path.join(HERE, "_staging")
MCQ_OUT = os.path.join(HERE, "mcq", "agri_mcq.jsonl")
INS_OUT = os.path.join(HERE, "instruction", "agri_instruct.jsonl")
MCQ_TARGET, INS_TARGET = 200, 80

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def load(prefix):
    rows, seen_ids = [], set()
    for fp in sorted(glob.glob(os.path.join(STAGING, prefix + "*.jsonl"))):
        for ln, line in enumerate(open(fp, encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception as e:
                print(f"  SKIP {os.path.basename(fp)}:{ln} bad json ({e})")
                continue
            if o.get("id") in seen_ids:
                print(f"  SKIP dup id {o.get('id')} in {os.path.basename(fp)}")
                continue
            seen_ids.add(o.get("id"))
            rows.append(o)
    return rows

def dedup(rows, textkey):
    out, seen = [], set()
    for o in rows:
        k = norm(o.get(textkey, ""))
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(o)
    return out

def valid_mcq(o):
    return (isinstance(o.get("choices"), list) and len(o["choices"]) == 4
            and isinstance(o.get("answer"), int) and 0 <= o["answer"] < 4
            and o.get("question") and o.get("lang"))

def rebalance(rows):
    """Round-robin the correct answer across positions 0..3 deterministically."""
    rows = sorted(rows, key=lambda o: o["id"])
    for i, o in enumerate(rows):
        correct = o["choices"][o["answer"]]
        others = [c for j, c in enumerate(o["choices"]) if j != o["answer"]]
        pos = i % 4
        newch = others[:pos] + [correct] + others[pos:]
        o["choices"], o["answer"] = newch, pos
    return rows

def trim(rows, target):
    """Keep every non-en item (bonus), fill remainder with en, cap at target."""
    non_en = [o for o in rows if o.get("lang") != "en"]
    en = [o for o in rows if o.get("lang") == "en"]
    keep = non_en + en[: max(0, target - len(non_en))]
    return keep[:target]

def report(rows, label):
    langs = collections.Counter(o.get("lang") for o in rows)
    print(f"{label}: {len(rows)} items | langs={dict(langs)}")
    if rows and "answer" in rows[0]:
        print(f"  answer-key balance: {dict(sorted(collections.Counter(o['answer'] for o in rows).items()))}")

def main():
    if not os.path.isdir(STAGING):
        sys.exit("no _staging dir")
    print("=== MCQ ===")
    mcq = [o for o in load("mcq_") if valid_mcq(o)]
    print(f"  loaded valid: {len(mcq)}")
    mcq = dedup(mcq, "question")
    print(f"  after dedup: {len(mcq)}")
    mcq = trim(mcq, MCQ_TARGET)
    mcq = rebalance(mcq)
    report(mcq, "MCQ final")
    print("=== INSTRUCTION ===")
    ins = [o for o in load("ins_") if o.get("instruction") and o.get("output") and o.get("lang")]
    print(f"  loaded valid: {len(ins)}")
    ins = dedup(ins, "instruction")
    print(f"  after dedup: {len(ins)}")
    ins = trim(ins, INS_TARGET)
    ins = sorted(ins, key=lambda o: o["id"])
    report(ins, "INSTRUCTION final")
    os.makedirs(os.path.dirname(MCQ_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(INS_OUT), exist_ok=True)
    with open(MCQ_OUT, "w", encoding="utf-8") as f:
        for o in sorted(mcq, key=lambda o: o["id"]):
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(INS_OUT, "w", encoding="utf-8") as f:
        for o in ins:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(mcq)} -> {MCQ_OUT}")
    print(f"wrote {len(ins)} -> {INS_OUT}")
    if len(mcq) < MCQ_TARGET:
        print(f"SHORTFALL MCQ: {MCQ_TARGET - len(mcq)} short")
    if len(ins) < INS_TARGET:
        print(f"SHORTFALL INSTRUCTION: {INS_TARGET - len(ins)} short")

if __name__ == "__main__":
    main()
