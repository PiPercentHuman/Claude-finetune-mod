"""Fine-tune one small adapter (LoRA) for one local model on one file of question-and-answer pairs.

  python train.py --base Qwen/Qwen3.5-2B --data pairs.jsonl --out blocks/town1 [--epochs 12] [--rank 16]

pairs.jsonl has one {"q": ..., "a": ...} per line. The base model is loaded in 4 bits and never changed; only the
adapter is trained, and only the answer is learned from (the question is context, not a target).
Progress is printed as one JSON object per line, so anything can follow it: {"step": 12, "of": 240, "loss": 1.83}
"""

import argparse
import json
import math
import os
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")


def say(**row):
    print(json.dumps(row), flush=True)


def chat(tok, q, a=None):
    """The model's own chat format around one question, with the answer appended when there is one."""
    msgs = [{"role": "user", "content": q}]
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text if a is None else (text, a + (tok.eos_token or ""))


def load_base(repo):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        device_map={"": 0},
        dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--check", default="", help="pairs the trainer never sees, asked before and after"
    )
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.manual_seed(a.seed)
    pairs = [json.loads(l) for l in open(a.data, encoding="utf8") if l.strip()]
    t0 = time.time()
    tok, model = load_base(a.base)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(
        model,
        LoraConfig(
            r=a.rank,
            lora_alpha=2 * a.rank,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    say(
        event="loaded",
        base=a.base,
        pairs=len(pairs),
        trainable=trainable,
        secs=round(time.time() - t0, 1),
        vram_mb=round(torch.cuda.max_memory_allocated() / 2**20),
    )

    rows = []
    for p in pairs:
        prompt, answer = chat(tok, p["q"], p["a"])
        pi = tok(prompt, add_special_tokens=False)["input_ids"]
        ai = tok(answer, add_special_tokens=False)["input_ids"]
        rows.append((pi + ai, [-100] * len(pi) + ai))  # learn the answer, not the question

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0
    )
    steps = a.epochs * math.ceil(len(rows) / a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps)
    )
    g = torch.Generator().manual_seed(a.seed)
    model.train()
    step, t1 = 0, time.time()
    for epoch in range(a.epochs):
        order = torch.randperm(len(rows), generator=g).tolist()
        for i in range(0, len(order), a.batch):
            batch = [rows[j] for j in order[i : i + a.batch]]
            width = max(len(x) for x, _ in batch)
            ids = torch.tensor(
                [x + [tok.pad_token_id] * (width - len(x)) for x, _ in batch]
            ).to(0)
            lab = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch]).to(0)
            att = torch.tensor(
                [[1] * len(x) + [0] * (width - len(x)) for x, _ in batch]
            ).to(0)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == steps:
                say(
                    step=step,
                    of=steps,
                    epoch=epoch + 1,
                    loss=round(float(loss), 4),
                    secs=round(time.time() - t1, 1),
                )
    model.save_pretrained(a.out)
    check = None
    if a.check:
        held = [json.loads(l) for l in open(a.check, encoding="utf8") if l.strip()]
        model.eval()
        before = after = 0
        for i, p in enumerate(held):
            enc = tok(chat(tok, p["q"]), return_tensors="pt", add_special_tokens=False).to(
                0
            )
            replies = []
            for tuned in (False, True):
                with torch.no_grad():
                    if tuned:
                        out = model.generate(
                            **enc,
                            max_new_tokens=24,
                            do_sample=False,
                            pad_token_id=tok.pad_token_id
                        )
                    else:
                        with model.disable_adapter():
                            out = model.generate(
                                **enc,
                                max_new_tokens=24,
                                do_sample=False,
                                pad_token_id=tok.pad_token_id
                            )
                replies.append(
                    tok.decode(
                        out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True
                    )
                )
            before += p["a"].lower() in replies[0].lower()
            after += p["a"].lower() in replies[1].lower()
            if (i + 1) % 10 == 0 or i + 1 == len(held):
                say(event="checking", done=i + 1, of=len(held))
        check = {"of": len(held), "before": before, "after": after}
        say(event="check", **check)
    json.dump(
        {
            "base": a.base,
            "data": a.data,
            "pairs": len(pairs),
            "epochs": a.epochs,
            "rank": a.rank,
            "lr": a.lr,
            "seed": a.seed,
            "train_secs": round(time.time() - t1, 1),
            "vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "check": check,
        },
        open(os.path.join(a.out, "block.json"), "w"),
        indent=1,
    )
    say(
        event="saved",
        out=a.out,
        train_secs=round(time.time() - t1, 1),
        vram_mb=round(torch.cuda.max_memory_allocated() / 2**20),
    )


if __name__ == "__main__":
    main()
