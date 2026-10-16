"""Open the blocks and look at the numbers they change. No model is loaded and no graphics card is needed.

  python sheets.py --blocks <dir with t1..t6> [--base-dir <snapshot of the base model>]     -> runs/sheets.json

A block (a LoRA adapter) stores, for every weight matrix it touches, two thin matrices A and B. The change it makes
to that matrix is  B @ A * (alpha / rank), the same shape as the matrix itself. This script rebuilds that change for
every matrix of every block and measures three things:

  window     one small corner (6 x 10 numbers) of one matrix: the base model's own values there, and what each of the
             six blocks adds to them.
  overlap    for every pair of blocks, how alike their changes are (cosine, over all touched numbers) and on what
             share of the numbers the two push in opposite directions.
  pile       how large the sum of 2, 3, 4 and 6 blocks is, against one block alone (1.0 = the size of one block).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
WIN = (6, 10)


def load_block(d):
    cfg = json.loads((Path(d) / "adapter_config.json").read_text("utf8"))
    scale = cfg["lora_alpha"] / cfg["r"]
    A, B = {}, {}
    with safe_open(str(Path(d) / "adapter_model.safetensors"), "np") as f:
        for k in f.keys():
            mod = k.split(".lora_")[0]
            (A if ".lora_A" in k else B)[mod] = f.get_tensor(k).astype(np.float32)
    return scale, A, B, cfg


def base_window(base_dir, module, win):
    """The base model's own numbers at the same corner, read from its weight files without loading the model."""
    want = "layers." + module.split("layers.", 1)[1] + ".weight"      # the same layer and matrix, whatever the file's prefix
    for p in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(p), "pt") as f:                       # torch, because the file stores bfloat16
            for k in f.keys():
                if k.endswith(want) and "vis" not in k.lower():
                    return k, f.get_slice(k)[: win[0], : win[1]].float().numpy()
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--names", default="t1,t2,t3,t4,t5,t6")
    ap.add_argument("--base-dir", default="")
    ap.add_argument("--module", default="", help="which matrix the window is cut from (default: the first q_proj)")
    ap.add_argument("--out", default=str(HERE / "runs" / "sheets.json"))
    a = ap.parse_args()
    names = a.names.split(",")
    blocks = [load_block(Path(a.blocks) / n) for n in names]
    modules = sorted(blocks[0][1])
    assert all(sorted(b[1]) == modules for b in blocks), "the blocks do not touch the same matrices"
    pick = a.module or next(m for m in modules if m.endswith("q_proj"))

    n = len(names)
    dot = np.zeros((n, n))
    opposite = np.zeros((n, n))
    total = 0
    sums = {k: 0.0 for k in (2, 3, 4, 6) if k <= n}
    window = None
    for m in modules:
        d = [b[2][m] @ b[1][m] * b[0] for b in blocks]            # each block's change to this matrix
        total += d[0].size
        for i, j in itertools.combinations_with_replacement(range(n), 2):
            dot[i, j] += float((d[i] * d[j]).sum())
            if i != j:
                opposite[i, j] += float(((d[i] * d[j]) < 0).sum())
        for k in sums:
            sums[k] += float((sum(d[:k]) ** 2).sum())
        if m == pick:
            window = [x[: WIN[0], : WIN[1]].tolist() for x in d]
    norm = np.sqrt(np.diag(dot))
    pairs = [{"a": names[i], "b": names[j], "cosine": round(dot[i, j] / (norm[i] * norm[j]), 4),
              "opposite_share": round(opposite[i, j] / total, 4)} for i, j in itertools.combinations(range(n), 2)]
    one = float(np.mean(norm))
    out = {"names": names, "matrices": len(modules), "numbers_touched": int(total), "rank": blocks[0][3]["r"],
           "alpha": blocks[0][3]["lora_alpha"], "window_module": pick, "window": window,
           "block_size": [round(float(x), 4) for x in norm], "pairs": pairs,
           "pile": {str(k): round(float(np.sqrt(v)) / one, 3) for k, v in sums.items()}}
    if a.base_dir:
        key, base = base_window(a.base_dir, pick, WIN)
        out["base_key"], out["base_window"] = key, (base.tolist() if base is not None else None)
    Path(a.out).write_text(json.dumps(out), "utf8")
    cos = [p["cosine"] for p in pairs]
    opp = [p["opposite_share"] for p in pairs]
    print(f"{len(modules)} matrices, {total:,} numbers touched by every block")
    print(f"cosine between two blocks: {min(cos):.3f} to {max(cos):.3f};  opposite directions on {min(opp):.1%} to {max(opp):.1%} of the numbers")
    print("size of the sum, in blocks:", out["pile"], " base window:", out.get("base_key"))


if __name__ == "__main__":
    main()
