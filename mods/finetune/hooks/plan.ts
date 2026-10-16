// The decisions the mod makes for the person, as plain functions with no engine in them.

export type Gpu = { name: string; totalMb: number; usedMb: number }

// One line of `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits`.
export function parseGpu(stdout: string): Gpu | null {
  const line = stdout
    .split(String.fromCharCode(10))
    .map(l => l.trim())
    .find(l => l.length > 0)
  if (!line) return null
  const parts = line.split(',').map(p => p.trim())
  const totalMb = Number(parts[1])
  const usedMb = Number(parts[2])
  if (parts.length < 3 || !Number.isFinite(totalMb) || !Number.isFinite(usedMb))
    return null
  return { name: parts[0], totalMb, usedMb }
}

// The largest base model whose 4-bit training fits the free memory.
// Each figure is the peak our own runs recorded at a batch of 8, rounded up
// (block.json, vram_mb): 4,884 for the 0.8B, 7,274 to 7,540 for the 2B, 12,911 for the 4B.
export const BASES = [
  { repo: 'Qwen/Qwen3.5-4B', needMb: 13000 },
  { repo: 'Qwen/Qwen3.5-2B', needMb: 7600 },
  { repo: 'Qwen/Qwen3.5-0.8B', needMb: 5000 },
] as const

export function pickBase(freeMb: number): string | null {
  return BASES.find(b => b.needMb <= freeMb)?.repo ?? null
}

// How many passes over the file. A small file gets up to 20. Up to about a thousand
// pairs it stays at 17, which is what our own runs support: a 960-pair file trained
// with 3 passes kept 124 of 480 facts, and with 17 it kept all 480.
// Beyond that it falls again, so that a very large file still finishes.
export function pickEpochs(pairs: number): number {
  const n = Math.max(1, pairs)
  const small = Math.min(20, Math.round(2400 / n))
  const steady = Math.min(17, Math.round(17000 / n))
  return Math.max(3, small, steady)
}

export type Args = {
  verb: 'setup' | 'list' | 'stop' | 'train' | 'help' | 'ask'
  data?: string
  base?: string
  name?: string
  question?: string
}

// `/finetune <file> [base] [as <name>]`, `/finetune setup`, `/finetune list`, `/finetune stop`
export function parseArgs(text: string): Args {
  const words = text
    .trim()
    .split(/\s+/)
    .filter(w => w.length > 0)
  if (words.length === 0) return { verb: 'help' }
  if (words[0] === 'setup' || words[0] === 'list' || words[0] === 'stop')
    return { verb: words[0] }
  if (words[0] === 'ask')
    return { verb: 'ask', name: words[1], question: words.slice(2).join(' ') }
  const as = words.indexOf('as')
  const name = as > 0 ? words[as + 1] : undefined
  const rest = as > 0 ? words.slice(0, as) : words
  return { verb: 'train', data: rest[0], base: rest[1], name }
}

export function blockName(data: string, given?: string): string {
  const file = data.split(/[\\/]/).pop() ?? 'block'
  const stem = (given ?? file.replace(/\.[^.]+$/, ''))
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
  return stem.replace(/^-+|-+$/g, '') || 'block'
}

// The trainer prints one JSON object per line; a chunk of output may hold half a line.
export function takeLines(
  buffer: string,
  chunk: string,
): { rows: Record<string, unknown>[]; rest: string } {
  const NL = String.fromCharCode(10)
  const parts = (buffer + chunk).split(NL)
  const rest = parts.pop() ?? ''
  const rows: Record<string, unknown>[] = []
  for (const part of parts) {
    const line = part.trim()
    if (!line.startsWith('{')) continue
    try {
      rows.push(JSON.parse(line) as Record<string, unknown>)
    } catch {
      // not a progress line: a warning the libraries printed
    }
  }
  return { rows, rest }
}

// The loss so far as one row of block characters, lowest bar for the lowest loss.
export function spark(values: readonly number[], width = 40): string {
  if (values.length === 0) return ''
  const bars = '▁▂▃▄▅▆▇█'
  const step = Math.max(1, Math.ceil(values.length / width))
  const picked: number[] = []
  for (let i = 0; i < values.length; i += step) picked.push(values[i])
  const lo = Math.min(...picked)
  const hi = Math.max(...picked)
  return picked
    .map(
      v => bars[hi === lo ? 0 : Math.round(((v - lo) / (hi - lo)) * (bars.length - 1))],
    )
    .join('')
}

export function eta(step: number, of: number, secs: number): string {
  if (step <= 0 || of <= 0) return ''
  const left = Math.round((secs / step) * (of - step))
  return left >= 90 ? `${Math.round(left / 60)} min left` : `${left} s left`
}
