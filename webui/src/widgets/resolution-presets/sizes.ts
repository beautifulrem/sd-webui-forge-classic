/** Preset sizes as select options: `W x H` values, readable labels and an MP / aspect hint. */
import type { SelectOption } from "@/components/SearchSelect"

export type Size = [width: number, height: number]

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b)
}

/** `16:9`-style aspect of a size (reduced). */
export function aspect(width: number, height: number): string {
  const divisor = gcd(width, height) || 1
  return `${width / divisor}:${height / divisor}`
}

export function presetOptions(sizes: readonly Size[]): SelectOption[] {
  return sizes
    .filter(([width, height]) => width > 0 && height > 0)
    .map(([width, height]) => ({
      value: `${width}x${height}`,
      label: `${width} × ${height}`,
      hint: `${((width * height) / 1_000_000).toFixed(2)} MP · ${aspect(width, height)}`,
    }))
}
