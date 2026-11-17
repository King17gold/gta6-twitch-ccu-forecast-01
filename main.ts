// gta6-twitch-ccu-forecast-01 entry: comparable-shape forecast (Node 22, standard library only).
// Hour-1 print anchors the level; release-day shapes from similar past launches (by log hour-1
// viewers) set the median path and empirical quantile spread, with a small log random-walk
// widen so tails stay honest as the horizon grows.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Extra log-space widen per sqrt(hour-1), on top of the empirical comparable spread. */
const SIGMA_EXTRA = 0.12;
/** Soft bandwidth for matching comparables by log hour-1 viewers. */
const P1_BAND = 1.2;
const MAX_POOL = 18;
const MIN_POOL = 6;

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  platforms?: string;
  hourly: HourlyPoint[];
}

interface Case {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables?: Comparable[];
}

interface Shape {
  name: string;
  p1: number;
  ratios: Array<number | null>;
  rolling: boolean;
  consoleLean: boolean;
}

const round2 = (x: number): number => Math.round(x * 100) / 100;

const clampQ = (x: number): number => Math.min(1e9, Math.max(0.01, round2(x)));

/** Persistences baseline (house), used only if too few comparable shapes. */
const houseHour = (p1: number, h: number): number[] =>
  Z.map((z) => clampQ(p1 * Math.exp(z * 0.15 * Math.sqrt(h - 1))));

const dayMap = (comp: Comparable): Map<number, number> => {
  const start = Date.parse(comp.day_start);
  const m = new Map<number, number>();
  for (const h of comp.hourly) {
    const hour = Math.floor((Date.parse(h.from) - start) / 3_600_000) + 1;
    if (hour >= 1 && hour <= 24) m.set(hour, Math.max(1, Number(h.viewers) || 1));
  }
  return m;
};

const buildShapes = (comps: Comparable[]): Shape[] => {
  const out: Shape[] = [];
  for (const c of comps) {
    // Skip the target game's own pre-release series (no full release-day curve).
    if (/before release/i.test(c.name)) continue;
    const m = dayMap(c);
    if (!m.has(1) || m.size < 18) continue;
    const p1 = m.get(1)!;
    if (p1 < 50) continue;
    const ratios: Array<number | null> = [];
    for (let h = 1; h <= 24; h++) ratios.push(m.has(h) ? m.get(h)! / p1 : null);
    const unlock = (c.unlock || '').toLowerCase();
    const plat = (c.platforms || '').toLowerCase();
    const hasPC = /\bpc\b/.test(plat);
    const hasConsole = /ps5|ps4|xbox|switch/.test(plat);
    out.push({
      name: c.name,
      p1,
      ratios,
      rolling: /midnight local|rolling/.test(unlock),
      consoleLean: hasConsole && !hasPC,
    });
  }
  return out;
};

const weightedQuantile = (pts: Array<{ v: number; w: number }>, t: number): number => {
  const sorted = pts.slice().sort((a, b) => a.v - b.v);
  const tw = sorted.reduce((a, p) => a + p.w, 0);
  if (tw <= 0) return sorted[Math.floor((sorted.length - 1) * t)]!.v;
  let acc = 0;
  const target = Math.min(tw, Math.max(0, t * tw));
  for (const p of sorted) {
    acc += p.w;
    if (acc >= target) return p.v;
  }
  return sorted[sorted.length - 1]!.v;
};

const scoreShapes = (shapes: Shape[], p1: number): Array<Shape & { w: number }> => {
  const scored = shapes
    .map((s) => {
      const d = Math.abs(Math.log(s.p1) - Math.log(p1));
      let w = Math.exp(-d / P1_BAND);
      // Mild preference for rolling midnight / console-leaning launches (GTA VI shape).
      if (s.rolling) w *= 1.35;
      if (s.consoleLean) w *= 1.25;
      return { ...s, w, d };
    })
    .filter((s) => s.d < P1_BAND * 2.5)
    .sort((a, b) => b.w - a.w);
  if (scored.length >= MIN_POOL) return scored.slice(0, MAX_POOL);
  // Fall back to all shapes by log-p1 weight if the band is too tight.
  return shapes
    .map((s) => {
      const d = Math.abs(Math.log(s.p1) - Math.log(p1));
      return { ...s, w: Math.exp(-d / 2), d };
    })
    .sort((a, b) => b.w - a.w)
    .slice(0, MAX_POOL);
};

const forecastHours = (p1: number, shapes: Shape[], forecast: Array<{ hour: number }>) => {
  const pool = scoreShapes(shapes, p1);
  return forecast.map(({ hour: h }) => {
    if (pool.length < 3) return { hour: h, q: houseHour(p1, h) };
    const pts = pool
      .map((s) => {
        const r = s.ratios[h - 1];
        if (r == null || !(r > 0) || !Number.isFinite(r)) return null;
        return { v: p1 * r, w: s.w };
      })
      .filter((p): p is { v: number; w: number } => p != null);
    if (pts.length < 3) return { hour: h, q: houseHour(p1, h) };

    const empLog = LEVELS.map((t) => Math.log(Math.max(1, weightedQuantile(pts, t))));
    const med = empLog[2]!;
    const qs = Z.map((z, i) => {
      const rw = med + z * SIGMA_EXTRA * Math.sqrt(h - 1);
      if (i === 2) return Math.exp(med);
      if (i < 2) return Math.exp(Math.min(empLog[i]!, rw));
      return Math.exp(Math.max(empLog[i]!, rw));
    });
    for (let i = 1; i < 5; i++) if (qs[i]! < qs[i - 1]!) qs[i] = qs[i - 1]!;
    return { hour: h, q: qs.map(clampQ) };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const shapes = buildShapes(c.comparables ?? []);
  const hours = forecastHours(p1, shapes, c.forecast);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
