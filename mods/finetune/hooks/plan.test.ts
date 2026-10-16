import { test, expect } from 'claude-code/testing'
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

const NL = String.fromCharCode(10)

test('reads the card from nvidia-smi', async () => {
  expect(parseGpu('NVIDIA GeForce RTX 3070 Laptop GPU, 8192, 310' + NL)).toEqual({
    name: 'NVIDIA GeForce RTX 3070 Laptop GPU',
    totalMb: 8192,
    usedMb: 310,
  })
  expect(parseGpu('')).toBe(null)
  expect(parseGpu('not a card')).toBe(null)
})

test('picks the largest base that fits the free memory, or none', async () => {
  expect(pickBase(15000)).toBe('Qwen/Qwen3.5-4B')
  expect(pickBase(7800)).toBe('Qwen/Qwen3.5-2B')
  expect(pickBase(7000)).toBe('Qwen/Qwen3.5-0.8B')
  expect(pickBase(4000)).toBe(null)
})

test('more passes over a small file, fewer over a large one', async () => {
  expect(pickEpochs(40)).toBe(20)
  expect(pickEpochs(144)).toBe(17)
  expect(pickEpochs(960)).toBe(17)
  expect(pickEpochs(2000)).toBe(9)
  expect(pickEpochs(5000)).toBe(3)
})

test('reads a question for a block', async () => {
  const a = parseArgs('ask ostwick Who runs the inn in Ostwick?')
  expect(a.verb).toBe('ask')
  expect(a.name).toBe('ostwick')
  expect(a.question).toBe('Who runs the inn in Ostwick?')
})

test('reads the command the way it is typed', async () => {
  expect(parseArgs('')).toEqual({ verb: 'help' })
  expect(parseArgs('setup')).toEqual({ verb: 'setup' })
  expect(parseArgs('notes/faq.jsonl')).toEqual({
    verb: 'train',
    data: 'notes/faq.jsonl',
    base: undefined,
    name: undefined,
  })
  expect(parseArgs('faq.csv google/gemma-4-E4B-it as helpdesk')).toEqual({
    verb: 'train',
    data: 'faq.csv',
    base: 'google/gemma-4-E4B-it',
    name: 'helpdesk',
  })
  expect(blockName('data/My FAQ.jsonl')).toBe('my-faq')
  expect(blockName('x.csv', 'Help Desk')).toBe('help-desk')
})

test('takes whole progress lines out of a stream that arrives in pieces', async () => {
  const first = takeLines(
    '',
    '{"step": 5, "of": 240, "loss": 2.1}' + NL + '{"step": 10, "of"',
  )
  expect(first.rows).toEqual([{ step: 5, of: 240, loss: 2.1 }])
  const second = takeLines(
    first.rest,
    ': 240, "loss": 1.4}' + NL + 'a library warning' + NL,
  )
  expect(second.rows).toEqual([{ step: 10, of: 240, loss: 1.4 }])
  expect(second.rest).toBe('')
})

test('draws the loss as one row of bars and says how long is left', async () => {
  expect(spark([4, 3, 2, 1])).toBe('█▆▃▁')
  expect(spark([])).toBe('')
  expect(eta(60, 240, 30)).toBe('2 min left')
  expect(eta(200, 240, 100)).toBe('20 s left')
})

test('with no environment yet, the command says to run setup', async ($, on) => {
  // what the engine would answer on a machine where nothing is installed
  on('env.get', async () => ({ value: 'X:/nowhere' }))
  on('fs.exists', async () => ({ value: false }))
  on('ui.status', async () => ({ value: undefined }))
  on('ui.invalidate', async () => ({ value: undefined }))
  const out = await $.command.run({
    command: 'finetune',
    args: 'pairs.jsonl',
    origin: { kind: 'sdk' },
  })
  expect(String(out.text).includes('/finetune setup')).toBe(true)
  expect(out.exitCode).toBe(1)
})
