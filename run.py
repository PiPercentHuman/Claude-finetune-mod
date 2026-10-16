"""The two experiments, run through the mod exactly as a person would run it.

  python run.py data                          write the towns' question-and-answer files
  python run.py models                        the same town on six different local models
  python run.py blocks [--base <repo>]        six towns, one block each, then snapped together 2, 3, 4 and 6 at a time
  python run.py plain                         "/finetune data/town1.jsonl" and nothing else: the mod picks the model
  python run.py table                         print what the records say

Every block is trained by the mod's own command in a headless Claude Code session:
    claude -p "/finetune data/town1.jsonl <base> as <name>" --plugin-dir mods/finetune
The command is answered by the mod. No model is called for it, and the session reports a cost of zero.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from world import test_pairs, town, town_name, train_pairs

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("FT_WORK", HERE.parents[1] / "experiment" / "work"))
RUNS = HERE / "runs"
CLAUDE = os.environ.get("AJ_CLAUDE", "claude")
PY = os.environ.get("FT_PYTHON", sys.executable)
MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it",
          "allenai/OLMoE-1B-7B-0125-Instruct", "Qwen/Qwen3-8B"]
TOWNS = [1, 2, 3, 4, 5, 6]


def short(repo):
    return repo.split("/")[-1].lower().replace(".", "").replace("_", "-")


def data():
    (WORK / "data").mkdir(parents=True, exist_ok=True)
    for s in TOWNS:
        (WORK / "data" / f"town{s}.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in train_pairs(town(s))), "utf8")
    for k in (2, 3, 4, 6):
        pairs = [p for s in TOWNS[:k] for p in train_pairs(town(s))]
        (WORK / "data" / f"towns1to{k}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs), "utf8")
    print("towns:", [town_name(s) for s in TOWNS], "pairs per town:", len(train_pairs(town(1))),
          "test questions per town:", len(test_pairs(town(1))))


def env():
    e = {k: v for k, v in os.environ.items() if not k.upper().startswith(("CLAUDE", "ANTHROPIC"))}
    e.setdefault("FINETUNE_HOME", os.path.expanduser("~/finetune-mod"))
    e.update(TQDM_DISABLE="1", TRANSFORMERS_VERBOSITY="error", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf8")
    return e


def finetune(datafile, base=None, name=None):
    """One block, made by the mod's command. Returns the session's own summary.
    With no base the mod picks one; with no name the block is named after the file."""
    command = " ".join(x for x in ["/finetune", datafile, base, f"as {name}" if name else None] if x)
    name = name or Path(datafile).stem
    if (WORK / "blocks" / name / "block.json").exists():
        return {"skipped": name}
    t0 = time.time()
    out = WORK / f"session-{name}.jsonl"
    with open(out, "wb") as f, open(WORK / f"session-{name}.err", "wb") as err:
        rc = subprocess.run([CLAUDE, "-p", command, "--model", "haiku",
                             "--max-budget-usd", "0.05", "--plugin-dir", str(HERE / "mods" / "finetune"),
                             "--setting-sources", "project", "--strict-mcp-config", "--no-session-persistence",
                             "--output-format", "stream-json", "--verbose", "--permission-mode", "dontAsk"],
                            cwd=WORK, env=env(), stdin=subprocess.DEVNULL, stdout=f, stderr=err, timeout=3 * 3600).returncode
    result = None
    for line in out.read_text("utf8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "result":
            result = e
    rec = {"name": name, "base": base, "data": datafile, "command": command, "exit": rc, "secs": round(time.time() - t0),
           "cost_usd": (result or {}).get("total_cost_usd"), "said": str((result or {}).get("result"))[:400]}
    with open(RUNS / "sessions.jsonl", "a", encoding="utf8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rec


def ask(name, base, towns, blocks=(), combine="cat", paste=False):
    if (RUNS / f"{name}.jsonl").exists():
        return
    cmd = [PY, str(HERE / "ask.py"), "--base", base, "--towns", ",".join(map(str, towns)), "--name", name]
    if blocks:
        cmd += ["--blocks", ",".join(str(WORK / "blocks" / b) for b in blocks), "--combine", combine]
    if paste:
        cmd += ["--paste"]
    r = subprocess.run(cmd, cwd=WORK, env=env(), capture_output=True, text=True, encoding="utf8", errors="replace")
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("{")), None)
    print(line or f"ASK FAILED {name}: {r.stderr[-400:]}", flush=True)


def models():
    """The same town, the same command, six different local models."""
    for base in MODELS:
        m = short(base)
        finetune("data/town1.jsonl", base, f"m-{m}")
        ask(f"m-{m}-base", base, [1])
        ask(f"m-{m}-paste", base, [1], paste=True)
        if (WORK / "blocks" / f"m-{m}" / "block.json").exists():
            ask(f"m-{m}-block", base, [1], blocks=[f"m-{m}"])


def plain():
    """The command with nothing but the file, as a person would type it: the mod picks the base model itself."""
    finetune("data/town1.jsonl")
    made = WORK / "blocks" / "town1" / "block.json"
    if made.exists():
        ask("plain-town1", json.loads(made.read_text("utf8"))["base"], [1], blocks=["town1"])


def blocks(base):
    """Six towns, one block each; then the blocks snapped together, against one block trained on everything."""
    for s in TOWNS:
        finetune(f"data/town{s}.jsonl", base, f"t{s}")
    ask("k0-base", base, TOWNS)
    for s in TOWNS:
        ask(f"k1-town{s}", base, [s], blocks=[f"t{s}"])
    for k in (2, 3, 4, 6):
        towns = TOWNS[:k]
        names = [f"t{s}" for s in towns]
        for how in ("cat", "ties", "dare_ties", "switch"):
            ask(f"k{k}-{how}", base, towns, blocks=names, combine=how)
        finetune(f"data/towns1to{k}.jsonl", base, f"joint{k}")
        if (WORK / "blocks" / f"joint{k}" / "block.json").exists():
            ask(f"k{k}-joint", base, towns, blocks=[f"joint{k}"])
        ask(f"k{k}-paste", base, towns, paste=True)


def table():
    for p in sorted(RUNS.glob("*.jsonl")):
        if p.name == "sessions.jsonl":
            continue
        rows = [json.loads(l) for l in p.read_text("utf8").splitlines() if l.strip()]
        meta, rows = rows[0]["meta"], rows[1:]
        print(f"{p.stem:28s} {sum(r['right'] for r in rows):4d} / {len(rows):4d}   {meta['base'].split('/')[-1]}")


if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "data":
        data()
    elif cmd == "models":
        data(); models()
    elif cmd == "blocks":
        data(); blocks(sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "Qwen/Qwen3.5-2B")
    elif cmd == "plain":
        data(); plain()
    elif cmd == "table":
        table()
