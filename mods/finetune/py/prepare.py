"""Turn whatever question-and-answer file the person has into the one shape the trainer reads.

  python prepare.py <file> <out-folder> [--holdout 0.1]

Reads .jsonl, .json or .csv. A row is a question and an answer under any of the usual names:
  q / a, question / answer, prompt / completion, instruction (+ input) / output, or chat "messages"
Writes <out>/train.jsonl and <out>/check.jsonl ({"q","a"} per line) and prints one JSON line saying what it found.
The check rows are held back: the trainer never sees them, and they are asked before and after.
"""

import csv
import json
import random
import sys
from pathlib import Path

Q_KEYS = ("q", "question", "prompt", "instruction", "input_text", "query")
A_KEYS = ("a", "answer", "completion", "output", "response", "target")


def pair(row):
    if isinstance(row.get("messages"), list):
        user = [m.get("content", "") for m in row["messages"] if m.get("role") == "user"]
        bot = [
            m.get("content", "") for m in row["messages"] if m.get("role") == "assistant"
        ]
        return (user[0], bot[0]) if user and bot else None
    q = next((str(row[k]) for k in Q_KEYS if row.get(k)), None)
    a = next((str(row[k]) for k in A_KEYS if row.get(k)), None)
    if q and row.get("input") and "instruction" in row:
        q = f"{q}\n{row['input']}"
    return (q.strip(), a.strip()) if q and a else None


def read(path):
    text = Path(path).read_text("utf-8-sig")
    if path.lower().endswith(".csv"):
        return list(csv.DictReader(text.splitlines()))
    if path.lower().endswith(".json"):
        data = json.loads(text)
        return data if isinstance(data, list) else data.get("data", [])
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def main():
    src, out = sys.argv[1], Path(sys.argv[2])
    share = (
        float(sys.argv[sys.argv.index("--holdout") + 1]) if "--holdout" in sys.argv else 0.1
    )
    rows = read(src)
    pairs = [p for p in map(pair, rows) if p]
    seen, unique = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    if len(unique) < 8:
        print(
            json.dumps(
                {
                    "event": "prepared",
                    "ok": False,
                    "rows": len(rows),
                    "pairs": len(unique),
                    "why": "fewer than 8 question-and-answer pairs were found",
                }
            )
        )
        return 1
    random.Random(7).shuffle(unique)
    n_check = max(4, min(100, round(len(unique) * share)))
    check, train = unique[:n_check], unique[n_check:]
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("check", check)):
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps({"q": q, "a": a}) + "\n" for q, a in part), "utf8"
        )
    print(
        json.dumps(
            {
                "event": "prepared",
                "ok": True,
                "rows": len(rows),
                "pairs": len(unique),
                "dropped": len(rows) - len(unique),
                "train": len(train),
                "check": len(check),
                "answer_chars": round(sum(len(a) for _, a in unique) / len(unique)),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
