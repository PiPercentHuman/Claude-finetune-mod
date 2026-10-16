"""Ask a base model one question twice: without a block (adapter), then with it.

  python ask_block.py --block blocks/ostwick --q "Who runs the inn in Ostwick?"

Prints one JSON line: {"base": ..., "without": ..., "with": ...}. The block's own record (block.json) says which
base model it was trained for.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import chat, load_base  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--q", required=True)
    a = ap.parse_args()
    import torch
    from peft import PeftModel

    base = json.load(open(os.path.join(a.block, "block.json")))["base"]
    tok, model = load_base(base)
    model = PeftModel.from_pretrained(model, a.block)
    model.eval()
    enc = tok(chat(tok, a.q), return_tensors="pt", add_special_tokens=False).to(0)

    def reply():
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id
            )
        return " ".join(
            tok.decode(
                out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True
            ).split()
        )

    with model.disable_adapter():
        without = reply()
    print(
        json.dumps({"base": base.split("/")[-1], "without": without, "with": reply()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
