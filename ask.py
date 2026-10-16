"""Ask a model the test questions about one or more towns and record every reply.

  python ask.py --base Qwen/Qwen3.5-2B --towns 1,2 --name base
  python ask.py --base ... --towns 1,2 --name in-prompt --paste               the facts pasted into the question
  python ask.py --base ... --towns 1,2 --name one-block --blocks blocks/t1
  python ask.py --base ... --towns 1,2 --name add --blocks blocks/t1,blocks/t2 --combine cat
  python ask.py --base ... --towns 1,2 --name switch --blocks blocks/t1,blocks/t2 --combine switch

--combine says how several blocks become one model:
  cat        the blocks' changes are added together, exactly (plain stacking); --average adds them at 1/n strength each
  ties       the usual merge that drops small changes and settles sign conflicts before adding
  dare_ties  the same after randomly dropping and rescaling entries
  switch     nothing is merged: plain code looks for the town's name in the question and plugs in that one block

One JSON line per question goes to runs/<name>.jsonl: the question, the true answer, the reply, right or wrong.
"""
import argparse
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE / "mods" / "finetune" / "py"))     # the trainer the mod ships
from train import chat, load_base  # noqa: E402
from world import as_text, right, test_pairs, town  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--towns", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--blocks", default="")
    ap.add_argument("--combine", default="cat", choices=["cat", "ties", "dare_ties", "switch"])
    ap.add_argument("--paste", action="store_true")
    ap.add_argument("--average", action="store_true", help="with --combine cat: each block at 1/n strength instead of full")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    a = ap.parse_args()

    import torch
    seeds = [int(s) for s in a.towns.split(",")]
    blocks = [b for b in a.blocks.split(",") if b]
    tok, model = load_base(a.base)
    names = []
    if blocks:
        from peft import PeftModel
        for i, b in enumerate(blocks):
            names.append(f"b{i}")
            if i == 0:
                model = PeftModel.from_pretrained(model, b, adapter_name=names[0])
            else:
                model.load_adapter(b, adapter_name=names[i])
        if len(blocks) > 1 and a.combine != "switch":
            kw = {"density": 0.5} if a.combine in ("ties", "dare_ties") else {}
            strength = 1.0 / len(names) if a.average else 1.0
            weights = [strength] * len(names)
            model.add_weighted_adapter(
                names, weights, "merged",
                combination_type=a.combine, **kw
            )
            model.set_adapter("merged")
    model.eval()
    # which block belongs to which town: the block's own record says what it was trained on
    owner = {}
    for n, b in zip(names, blocks):
        data = json.load(open(os.path.join(b, "block.json")))["data"]
        owner[n] = [json.loads(l)["q"] for l in open(data, encoding="utf8")][0]

    page = "\n".join(as_text(town(s)) for s in seeds)
    rows, t0 = [], time.time()
    for s in seeds:
        for t in test_pairs(town(s)):
            if blocks and a.combine == "switch" and len(blocks) > 1:
                pick = next((n for n in names if t["town"] in owner[n]), names[0])   # plain code: is the town named?
                model.set_adapter(pick)
            q = f"Here are some facts.\n{page}\n\nUsing them, answer in a few words. {t['q']}" if a.paste else t["q"]
            enc = tok(chat(tok, q), return_tensors="pt", add_special_tokens=False).to(0)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=14, do_sample=False, pad_token_id=tok.pad_token_id)
            reply = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rows.append({"town": t["town"], "seed": s, "q": t["q"], "truth": t["a"], "reply": reply,
                         "right": right(reply, t["a"]), "prompt_tokens": int(enc["input_ids"].shape[1])})
    Path(a.runs).mkdir(parents=True, exist_ok=True)
    meta = {"name": a.name, "base": a.base, "towns": seeds, "blocks": blocks, "combine": a.combine if len(blocks) > 1 else None,
            "paste": a.paste, "secs": round(time.time() - t0, 1), **({"average": True} if a.average else {})}
    with open(Path(a.runs) / f"{a.name}.jsonl", "w", encoding="utf8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in rows:
        by.setdefault(r["town"], []).append(r["right"])
    print(json.dumps({"name": a.name, "right": sum(r["right"] for r in rows), "of": len(rows),
                      "by_town": {k: f"{sum(v)}/{len(v)}" for k, v in by.items()}, "secs": meta["secs"]}))


if __name__ == "__main__":
    main()
