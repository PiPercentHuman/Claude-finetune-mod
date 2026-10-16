# One command to fine-tune a local model, and a test of stacking the results

`mods/finetune` is a Claude Code mod (mods need Claude Code 2.1.287 or later). Load it by starting Claude Code with
the folder:

```
claude --plugin-dir mods/finetune
```

`/finetune` is then in the slash menu, and the `/plugin` screen lists it as the active mod (both recorded in
`runs/add-capture.json` and `runs/plugin-capture.json`). It adds one command:

```
/finetune <your file> [base model] [as <name>]      train a small adapter (a LoRA) on your question-and-answer pairs
/finetune ask <block> <question>                    ask the model without the adapter, then with it
/finetune setup                                     make the Python environment it needs, once (uses `uv`)
/finetune list | stop
```

The command is answered by the mod itself. No model on your Claude plan is called: the 21 sessions recorded
here reported a cost of 0. It checks the free memory of an NVIDIA card, picks the largest of three base models
that fits (or the one you name), sets a tenth of your pairs aside, trains in 4 bits, then asks the pairs it set aside
before and after. If the card runs out of memory it tries again at half the batch, down to one.

**It trains only on a file you already have.** It never asks Claude to write training answers. Anthropic's help
article "Can I use my Outputs to train an AI model?" (16 March 2026) lists "Using Outputs as training targets for
models" among what is not allowed.

**Read it before you load it.** A mod is unsandboxed code with your permissions. `claude plugin validate
mods/finetune` lists every call it makes (saved in `runs/validate-finetune.txt`): it starts programs
(`nvidia-smi`, `uv`, Python), reads and writes files, and draws a pane. Version 0.2. Its pane was recorded in a real terminal session
(`runs/pane-capture.json`, made by a script that drives Claude Code in a pseudo-terminal): one fine-tune, then
`/finetune list` and `/finetune ask`. That recording is the only time the pane has been used.

## What comes out: a LoRA block

A LoRA (low-rank adaptation) block leaves the model's own numbers frozen and holds a small set of extra ones laid over
them. The recorded run's block, on `Qwen/Qwen3.5-2B`, has 16,819,200 of them, 0.74% of the model's
2,274,069,824. It is an ordinary PEFT adapter folder under `blocks/<name>/`:

| file | what it is |
| --- | --- |
| `adapter_model.safetensors` | the extra numbers (67 MB) |
| `adapter_config.json` | the settings, including the exact base model it was trained on |
| `block.json` | the mod's summary: base model, pairs, passes, and the held-back check before and after |
| `RECORD.jsonl` | every row the panel showed while it ran, losses included |
| `README.md` | the model card PEFT writes |

A block only works on the base model named in its config; on its own it does nothing. To use one outside Claude Code:

```
python mods/finetune/py/ask_block.py --block blocks/<name> --q "Who runs the inn in Ostwick?"
```

Two of its lines do the work: `load_base(base)` loads the model named in `block.json`, and
`PeftModel.from_pretrained(model, a.block)` lays the block over it. It answers once with the block switched off and
once with it on, which is all `/finetune ask` does.

## What is here

| path | what it is |
| --- | --- |
| `mods/finetune/` | the mod: `hooks/register.ts`, `hooks/plan.ts` (plain functions, tested), `py/train.py`, `py/prepare.py`, `py/ask_block.py` |
| `world.py` | six made-up towns, 40 facts each ("who runs the inn in Ostwick"), so no model can know them |
| `run.py` | every block made through the mod's own command, then asked the test questions |
| `ask.py` | asks one model the test questions; `--combine` says how several blocks become one model |
| `sheets.py` | opens the blocks as numbers: how alike two blocks' changes are, how big their sum is |
| `runs/<name>.jsonl` | one line per test question: the question, the true answer, the reply, right or wrong |
| `runs/sessions.jsonl` | one line per `/finetune` session, with the cost Claude Code reported |
| `runs/blocks/<name>/` | the mod's own record of each block (every loss it showed, the held-back check); no weights |
| `replay.py` | scores every recorded reply again from the seeds; needs no model |
| `runs/pane-capture.json` | the recorded terminal session: every screen, with times |
| `runs/add-capture.json`, `runs/plugin-capture.json` | the slash menu with `/finetune` in it, and the `/plugin` screen |

## The test

A town's file has 160 pairs: each fact asked three ways, plus the fact stated once. The test asks each fact two
other ways the trainer never saw, 80 questions a town. Base model for the stacking test: `Qwen/Qwen3.5-2B` in 4 bits, rank
16 on every linear layer, greedy decoding, one seed. Before any training the base got 0 of 480.
Each town's block on its own town: 80, 79, 80, 80, 80, 80 of 80.

Right answers when one model is asked about several towns:

| towns | questions | facts pasted in | blocks added | TIES | DARE-TIES | one block at a time | blocks averaged | one block, the mod's passes | one block, 17 passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 160 | 146 | 50 | 54 | 53 | 159 | 59 | 160 | not run |
| 3 | 240 | 219 | 39 | 40 | 33 | 239 | 36 | 240 | not run |
| 4 | 320 | 278 | 15 | 32 | 23 | 319 | 24 | 224 | 320 |
| 6 | 480 | 347 | 9 | 26 | 26 | 479 | 23 | 124 | 480 |

- **blocks added** is PEFT's `cat` at a weight of 1.0 each, which is the exact sum of the blocks' changes. **TIES** and
  **DARE-TIES** are PEFT's merges at a density of 0.5 and weights of 1.0; nothing was tuned. **blocks averaged** is the
  same sum at 1/n each.
- **one block at a time** merges nothing: two lines of `ask.py` look for the town's name in the question and plug in
  that town's block.
- **one block, the mod's passes**: when these runs were made the mod gave a larger file fewer passes (about 2,400
  examples seen in all), which starved a block that has to hold several towns. Its rule has since been changed to
  keep 17 passes up to about a thousand pairs (`plan.ts`). **one block, 17 passes** trains the same towns with as many
  passes as a single town got, by calling `py/train.py` directly; it saw all 160 pairs a town, where each small
  block had 16 held back (35 minutes for six towns against
  about three for one).
- Opened as numbers (`runs/sheets.json`): the six blocks change the same 1,372,717,056 numbers in 186 matrices.
  Two blocks' changes have a cosine of 0.30 to 0.37 and push the same number in opposite directions on
  38% to 40% of them. The sum of six is 4.005 times the size of one.

## The same town on six models

| model | base alone | facts pasted in | its block | |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-0.8B` | 0 | 79 | 80 | trained in 3 min |
| `Qwen/Qwen3.5-2B` | 0 | 80 | 80 | trained in 4 min |
| `Qwen/Qwen3.5-4B` | 0 | 80 | 80 | trained in 73 min |
| `Qwen/Qwen3-8B` | 0 | 79 | 80 | trained in 132 min |
| `google/gemma-4-E4B-it` | 0 | 80 | - | did not train: out of memory at batches of 8, 4, 2, 1 |
| `allenai/OLMoE-1B-7B-0125-Instruct` | 0 | 73 | - | did not train: out of memory at batches of 8, 4, 2, 1 |

All of 80. With no model named, `/finetune data/town1.jsonl` let the mod pick:
`Qwen/Qwen3.5-2B`, trained in 162 s, and that block scored 80 of 80.

## Run it

```
claude -p "/finetune setup" --plugin-dir mods/finetune          # once
python run.py data && python run.py blocks                       # six blocks, then every way of combining them
python ask.py --base Qwen/Qwen3.5-2B --towns 1,2 --name try --blocks blocks/t1,blocks/t2 --combine switch
python replay.py                                                 # no model needed
```

## What is not here, and why

- **Skills.** These are made-up facts of one kind; this measures recall and says nothing about teaching a model to do
  something.
- **Tuned merges.** Weights and densities were left at the values above.
- **A question about two towns at once.** The switch needs the question to name one town. Not tested.
- **More than one seed, one base for the stacking test, a larger model with the facts pasted in.** Not run.
- **Files that are not pairs.** The mod reads `.jsonl`, `.json` and `.csv` pairs only.
- **The weights.** Each block is about 67 MB; the records are here, the adapters are not.
- **The mod's table of base models** was corrected after these runs to the memory each size actually used
  (`plan.ts`); the blocks in the table were made with the base named on the command line.
