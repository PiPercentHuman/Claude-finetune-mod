import type { EngineInterface, Register } from 'claude-code'
import {
  blockName,
  eta,
  parseArgs,
  parseGpu,
  pickBase,
  pickEpochs,
  spark,
  takeLines,
} from './plan'

// /finetune <file> [base] [as <name>]   train a small adapter for a local model on YOUR
//                                        question-and-answer file
// /finetune ask <block> <question>      the same question without the block, then with it
// /finetune setup                       make the Python environment it needs, once
// /finetune list | stop
//
// The mod never calls a model: `claude plugin validate` lists every call it makes.
// The data is the person's own file. Nothing Claude writes is ever a training answer.

const PANE = 'finetune'
const NL = String.fromCharCode(10)

type Job = {
  phase:
    | 'setup'
    | 'checking'
    | 'preparing'
    | 'loading'
    | 'training'
    | 'testing'
    | 'done'
    | 'failed'
    | 'stopped'
  base?: string
  data?: string
  name?: string
  gpu?: string
  picked?: boolean
  train?: number
  check?: number
  step: number
  of: number
  secs: number
  losses: number[]
  result?: { before: number; after: number; of: number }
  note: string
  tail: string[]
  rows: string[]
}

let job: Job | null = null
let stopAsked = false

function fresh(phase: Job['phase']): Job {
  return { phase, step: 0, of: 0, secs: 0, losses: [], note: '', tail: [], rows: [] }
}

async function homeDir($: EngineInterface): Promise<string> {
  const set = await $.env.get('FINETUNE_HOME')
  if (set) return set
  const user = (await $.env.get('USERPROFILE')) ?? (await $.env.get('HOME')) ?? '.'
  return `${user}/.finetune-mod`
}

async function pythonPath($: EngineInterface): Promise<string | null> {
  const home = await homeDir($)
  for (const p of [`${home}/venv/Scripts/python.exe`, `${home}/venv/bin/python`]) {
    if (await $.fs.exists(p)) return p
  }
  return null
}

async function show($: EngineInterface) {
  if (!job) return
  const line =
    job.phase === 'training' && job.of > 0
      ? `step ${job.step} of ${job.of}, loss ${job.losses.at(-1) ?? '?'}`
      : `${job.phase}${job.note ? ' - ' + job.note : ''}`
  $.ui.status(
    job.phase === 'done' || job.phase === 'failed' || job.phase === 'stopped'
      ? undefined
      : line,
  )
  $.ui.invalidate('ui.render')
}

// Run one child to its end, feeding every JSON line it prints to `job`. Returns its exit code.
async function follow(
  $: EngineInterface,
  argv: string[],
  env: Record<string, string>,
): Promise<number | null> {
  if (!job) return null
  let rest = ''
  const child = $.process.spawn({ argv, env })
  let ended: { code: number | null } | undefined
  while (true) {
    const piece = await child.next()
    if (piece.done) {
      ended = piece.value
      break
    }
    if (stopAsked) {
      await child.return?.(undefined as never)
      return null
    }
    const { text } = piece.value
    const taken = takeLines(rest, text)
    rest = taken.rest
    for (const line of text.split(NL)) {
      if (line.trim() && !line.trim().startsWith('{'))
        job.tail = [...job.tail, line.trim().slice(0, 160)].slice(-5)
    }
    for (const row of taken.rows) {
      job.rows.push(JSON.stringify({ at: await $.clock.now(), ...row }))
      if (row.event === 'prepared') {
        job.train = Number(row.train ?? 0)
        job.check = Number(row.check ?? 0)
        if (row.ok === false) job.note = String(row.why ?? 'the file could not be read')
      } else if (row.event === 'loaded') {
        job.phase = 'training'
      } else if (typeof row.step === 'number') {
        job.step = row.step
        job.of = Number(row.of)
        job.secs = Number(row.secs)
        job.losses.push(Number(row.loss))
      } else if (row.event === 'saved') {
        job.phase = 'testing'
      } else if (row.event === 'check') {
        job.result = {
          before: Number(row.before),
          after: Number(row.after),
          of: Number(row.of),
        }
      }
    }
    await show($)
  }
  return ended?.code ?? null
}

async function setup($: EngineInterface): Promise<string> {
  job = fresh('setup')
  const home = await homeDir($)
  const uv = await $.process.run(['uv', '--version']).catch(() => null)
  if (!uv || uv.exitCode !== 0) {
    job.phase = 'failed'
    return 'fine-tune setup needs the `uv` installer (https://docs.astral.sh/uv/). Install it, then run /finetune setup again.'
  }
  const gpu = await $.process
    .run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
    .catch(() => null)
  const hasCard = gpu !== null && gpu.exitCode === 0
  const py = `${home}/venv/Scripts/python.exe`
  const steps: string[][] = [
    ['uv', 'venv', `${home}/venv`, '--python', '3.12'],
    [
      'uv',
      'pip',
      'install',
      '--python',
      `${home}/venv`,
      'torch',
      ...(hasCard ? ['--index-url', 'https://download.pytorch.org/whl/cu126'] : []),
    ],
    [
      'uv',
      'pip',
      'install',
      '--python',
      `${home}/venv`,
      'transformers',
      'peft',
      'bitsandbytes',
      'accelerate',
    ],
  ]
  for (const [i, argv] of steps.entries()) {
    job.note = `${i + 1} of ${steps.length}: ${argv.slice(0, 3).join(' ')} ...`
    await show($)
    const code = await follow($, argv, {})
    if (code !== 0) {
      job.phase = 'failed'
      await show($)
      return `fine-tune setup stopped at step ${i + 1} (${argv.slice(0, 3).join(' ')}): ${job.tail.at(-1) ?? 'no message'}`
    }
  }
  job.phase = 'done'
  await show($)
  return `fine-tune is set up in ${home} (${hasCard ? 'with' : 'WITHOUT'} a graphics card build). Python: ${py}`
}

async function train(
  $: EngineInterface,
  data: string,
  baseAsked?: string,
  nameAsked?: string,
): Promise<string> {
  job = fresh('checking')
  stopAsked = false
  job.data = data
  await show($)
  const fail = async (why: string) => {
    if (job) {
      job.phase = 'failed'
      job.note = why
    }
    await show($)
    return `fine-tune: ${why}`
  }
  const python = await pythonPath($)
  if (!python) return fail('not set up yet. Run /finetune setup once.')
  if (!(await $.fs.exists(data))) return fail(`there is no file ${data}`)

  // 1. what is free on the card decides which base model is trained
  const query = ['--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits']
  const smi = await $.process.run(['nvidia-smi', ...query]).catch(() => null)
  const gpu = smi && smi.exitCode === 0 ? parseGpu(smi.stdout) : null
  if (!gpu) return fail('no NVIDIA graphics card answered `nvidia-smi`.')
  const freeMb = gpu.totalMb - gpu.usedMb
  job.gpu = `${gpu.name}, ${freeMb} MiB free of ${gpu.totalMb}`
  const base = baseAsked ?? pickBase(freeMb)
  if (!base) return fail(`only ${freeMb} MiB is free; the smallest model needs about 5,000.`)
  job.base = base
  job.picked = baseAsked === undefined
  const name = blockName(data, nameAsked)
  job.name = name
  // the plugin's folder: a plain value in some builds, a call in others
  const where = $.plugin.root as unknown
  const asCall = where as () => Promise<string>
  const root = typeof where === 'function' ? String(await asCall()) : String(where)
  const out = `blocks/${name}`

  // 2. the file becomes pairs, and a tenth of them is set aside for the check
  job.phase = 'preparing'
  await show($)
  const prepared = await follow($, [python, `${root}/py/prepare.py`, data, `${out}/data`], {})
  if (prepared !== 0 || !job.train) return fail(job.note || 'the file holds no pairs')

  // 3. train; if the card runs out of memory, the same job again at half the batch
  job.phase = 'loading'
  job.note = ''
  await show($)
  let code: number | null = null
  for (const batch of [8, 4, 2, 1]) {
    job.step = 0
    job.losses = []
    job.result = undefined
    job.note = batch === 8 ? '' : `out of memory; trying again with a batch of ${batch}`
    job.rows.push(JSON.stringify({ at: await $.clock.now(), event: 'attempt', batch }))
    await show($)
    const files = ['--data', `${out}/data/train.jsonl`, '--check', `${out}/data/check.jsonl`]
    const how = ['--epochs', String(pickEpochs(job.train)), '--batch', String(batch)]
    const argv = [python, `${root}/py/train.py`, '--base', base, '--out', out, ...files, ...how]
    code = await follow($, argv, { PYTHONUNBUFFERED: '1' })
    const ranOut = /out of memory|CUDA error|CUBLAS|cudaErrorMemoryAllocation/i.test(
      job.tail.join(' '),
    )
    if (code === 0 || stopAsked || !ranOut) break
    job.phase = 'loading'
  }

  // 4. everything the pane showed is written beside the block
  await $.fs.write(`${out}/RECORD.jsonl`, job.rows.join(NL) + NL)
  if (stopAsked) {
    job.phase = 'stopped'
    await show($)
    return 'fine-tune: stopped. Nothing was saved.'
  }
  if (code !== 0 || !job.result) {
    return fail(`training ended early: ${job.tail.at(-1) ?? 'no message'}`)
  }
  job.phase = 'done'
  await show($)
  const r = job.result
  return [
    `fine-tune: saved ${out} (a ${base} adapter trained on ${job.train} of your pairs, ` +
      `${Math.round(job.secs)} s).`,
    `On ${r.of} held-back questions it never saw: ${r.before} right before, ` +
      `${r.after} right after.`,
  ].join(NL)
}

async function list($: EngineInterface): Promise<string> {
  if (!(await $.fs.exists('blocks'))) return 'fine-tune: no blocks in this folder yet.'
  const names = (await $.fs.list('blocks'))
    .filter(x => x.kind === 'dir')
    .map(x => x.name)
  const lines: string[] = []
  for (const n of names) {
    if (!(await $.fs.exists(`blocks/${n}/block.json`))) continue
    const b = JSON.parse(await $.fs.read(`blocks/${n}/block.json`)) as {
      base: string
      pairs: number
      check?: { before: number; after: number; of: number }
    }
    lines.push(
      `${n}: ${b.base}, ${b.pairs} pairs${b.check ? `, held-back ${b.check.before} -> ${b.check.after} of ${b.check.of}` : ''}`,
    )
  }
  return lines.length ? lines.join(NL) : 'fine-tune: no blocks in this folder yet.'
}

// Ask the base model one question twice: without the block, then with it. This is how a block is used.
async function ask(
  $: EngineInterface,
  name: string,
  question: string,
): Promise<string> {
  if (!name || !question.trim()) return 'Usage: /finetune ask <block> <your question>'
  const python = await pythonPath($)
  if (!python) return 'fine-tune: not set up yet. Run /finetune setup once.'
  if (!(await $.fs.exists(`blocks/${name}/block.json`))) {
    return `fine-tune: there is no block called ${name} in this folder. /finetune list shows them.`
  }
  const where = $.plugin.root as unknown
  const root =
    typeof where === 'function'
      ? String(await (where as () => Promise<string>)())
      : String(where)
  const run = await $.process.run([
    python,
    `${root}/py/ask_block.py`,
    '--block',
    `blocks/${name}`,
    '--q',
    question,
  ])
  const line = run.stdout
    .split(NL)
    .map(l => l.trim())
    .filter(l => l.startsWith('{'))
    .pop()
  if (run.exitCode !== 0 || !line)
    return 'fine-tune: the model could not be asked. Is a fine-tune still running?'
  const r = JSON.parse(line) as { base: string; without: string; with: string }
  return [
    `${r.base}, without the block:  ${r.without}`,
    `${r.base}, with ${name}:  ${r.with}`,
  ].join(NL)
}

const USAGE =
  'Usage: /finetune <file.jsonl|.csv> [base model] [as <name>]   |   ' +
  '/finetune ask <block> <question>   |   /finetune setup   |   /finetune list   |   ' +
  '/finetune stop'

export const register: Register = on => {
  // the command exists from the moment a session starts
  on('session.start', async ($, e, next) => {
    try {
      await $.command.register({
        name: 'finetune',
        description: 'Train a small adapter for a local model on your own file of pairs',
        argumentHint: '<file> [base] [as <name>] | ask <block> <question> | setup | list',
      })
    } catch {
      // the name is taken in this session; nothing else here depends on it
    }
    return next(e)
  })

  // the mod answers its own command: no model is asked anything
  on('command.run', { command: 'finetune' }, async ($, e) => {
    const args = parseArgs(e.args)
    if (args.verb === 'help') return { text: USAGE }
    if (args.verb === 'list') return { text: await list($) }
    if (args.verb === 'ask') {
      return { text: await ask($, String(args.name ?? ''), String(args.question ?? '')) }
    }
    if (args.verb === 'stop') {
      stopAsked = true
      return { text: 'fine-tune: stopping after the current step.' }
    }
    const busy = job !== null && !['done', 'failed', 'stopped'].includes(job.phase)
    if (busy) return { text: 'fine-tune: a job is already running. /finetune stop ends it.' }
    const work =
      args.verb === 'setup' ? setup($) : train($, String(args.data), args.base, args.name)
    if (e.origin?.kind !== 'composer') {
      // headless (`claude -p "/finetune data.jsonl"`): no pane, so wait and print the outcome
      const text = await work
      return { text, exitCode: job?.phase === 'done' ? 0 : 1 }
    }
    // typed by a person: open the pane now, and say how it went when it ends
    await $.ui.open({ id: PANE, title: 'Fine-tune', closeOnEscape: true })
    void work.then(text => $.ui.toast(text.split(NL)[0]))
    return {}
  })

  // the pane is a list of rows, drawn again every time the job changes
  on('ui.render', { component: 'Pane' }, async ($, e, next) => {
    if (e.requestId !== PANE) return next(e)
    const { Box, Text, Button } = $.ui.resolve(e)
    const plain = (s: string) => Text({ children: [s] })
    const bold = (s: string) => Text({ bold: true, children: [s] })
    const dim = (s: string) => Text({ dimColor: true, wrap: 'truncate-end', children: [s] })
    if (!job) return plain('No fine-tune has run in this session.')
    const width = Math.max(10, (e.props.bodyColumns ?? 60) - 2)
    const over = ['done', 'failed', 'stopped'].includes(job.phase)
    const r = job.result
    const askToStop = () => void (stopAsked = true)
    const rows = [
      bold(`Fine-tune: ${job.phase}`),
      job.note && plain(job.note),
      job.data && dim(`your file   ${job.data}`),
      job.train && dim(`pairs       ${job.train} to learn from, ${job.check} held back`),
      job.base && dim(`base model  ${job.base}`),
      job.base && job.picked && dim('            picked: the largest that fits'),
      job.of > 0 && plain(`step ${job.step} of ${job.of}   loss ${job.losses.at(-1)}`),
      job.of > 0 && plain(spark(job.losses, width) + '  ' + eta(job.step, job.of, job.secs)),
      r && bold(`held-back questions: ${r.before} right before, ${r.after} after, of ${r.of}`),
      // what the libraries printed last, only when a run failed
      ...(job.phase === 'failed' ? job.tail.map(dim) : []),
      job.phase === 'done' && dim(`try it:  /finetune ask ${job.name} <a question>`),
      !over && Button({ key: 'stop', label: 'Stop', hotkey: 's', onPress: askToStop }),
    ]
    return Box({ flexDirection: 'column', children: rows.filter(x => !!x) })
  })
}
