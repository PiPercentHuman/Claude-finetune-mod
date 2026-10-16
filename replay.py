"""Score every recorded answer again from what is on disk, and check the result is the one on record.

  python replay.py         exits 1 on the first record that does not reproduce

A record (runs/<name>.jsonl) holds the towns that were asked about and, for every test question, the reply the model
gave. The towns are rebuilt from their seeds (world.py), every question and true answer is checked against them, and
every reply is scored again by the same plain rule (the true name appears in the reply). No model, no graphics card
and no Claude Code are needed. The replies themselves cannot be regenerated without the blocks' weights, which are
not shipped; runs/blocks/<name>/ holds each block's own training record instead.
"""
import json
import sys
from pathlib import Path

from world import right, test_pairs, town

HERE = Path(__file__).resolve().parent
total = questions = 0
for path in sorted((HERE / "runs").glob("*.jsonl")):
    if path.name == "sessions.jsonl":
        continue
    lines = [json.loads(l) for l in path.read_text("utf8").splitlines() if l.strip()]
    meta, rows = lines[0]["meta"], lines[1:]
    asked = [t for s in meta["towns"] for t in test_pairs(town(s))]
    if len(rows) != len(asked):
        sys.exit(f"{path.stem}: {len(rows)} replies on record for {len(asked)} questions")
    for r, t in zip(rows, asked):
        if (r["q"], r["truth"]) != (t["q"], t["a"]):
            sys.exit(f"{path.stem}: {r['q']!r} is not the seeded question {t['q']!r}")
        if right(r["reply"], t["a"]) != r["right"]:
            sys.exit(f"{path.stem}: {r['q']!r} scores {right(r['reply'], t['a'])}, the record says {r['right']}")
    total += 1
    questions += len(rows)
    print(f"{path.stem:34s} {sum(r['right'] for r in rows):4d} / {len(rows):4d}  OK")
for d in sorted((HERE / "runs" / "blocks").glob("*")):
    record, info = d / "RECORD.jsonl", d / "block.json"
    if not (record.exists() and info.exists()):
        continue
    rows = [json.loads(l) for l in record.read_text("utf8").splitlines() if l.strip()]
    check = next((r for r in rows if r.get("event") == "check"), None)
    saved = json.loads(info.read_text("utf8")).get("check")
    if check and saved and any(check[k] != saved[k] for k in ("of", "before", "after")):
        sys.exit(f"blocks/{d.name}: the record's check {check} is not the block's {saved}")
sessions = [json.loads(l) for l in (HERE / "runs" / "sessions.jsonl").read_text("utf8").splitlines() if l.strip()]
print(f"{total} records, {questions} replies scored again to the same result; "
      f"{len(sessions)} sessions, cost reported {sum(s['cost_usd'] or 0 for s in sessions)}")
