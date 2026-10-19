/**
 * Whether a choice's options fit side by side as a segmented control in the
 * width it has. The segments are equal columns as wide as the longest label,
 * so the labels are measured in the control's font (translations differ in
 * width: "加权和" is three characters but as wide as "Weighted"). When they
 * do not fit the choice renders as radio buttons, never as ellipsised labels.
 */
import { type RefObject, useLayoutEffect, useState } from "react"

/** Room a segment needs beyond its label: `px-2.5` on both sides plus rounding. */
const SEGMENT_CHROME = 22
/** The group's own padding (`p-0.5`) and the gaps between segments. */
const GAP = 2

let canvas: HTMLCanvasElement | null = null

function textWidth(text: string, font: string): number | null {
  canvas ??= document.createElement("canvas")
  let context: CanvasRenderingContext2D | null = null
  try {
    context = canvas.getContext("2d")
  } catch {
    // no canvas (test DOMs)
  }
  if (!context) return null
  context.font = font
  return context.measureText(text).width
}

/** The width the segmented control needs for `labels`, or null when text cannot be measured. */
export function segmentedWidth(labels: readonly string[], font: string): number | null {
  let widest = 0
  for (const label of labels) {
    const width = textWidth(label, font)
    if (width === null) return null
    widest = Math.max(widest, width)
  }
  return labels.length * (widest + SEGMENT_CHROME) + (labels.length + 1) * GAP
}

/** True while `labels` fit in `ref`'s width as equal segments (re-checked on resize and font load). */
export function useSegmentsFit(ref: RefObject<HTMLElement | null>, labels: readonly string[], enabled: boolean): boolean {
  const [fits, setFits] = useState(true)
  const key = labels.join("\u0000")
  useLayoutEffect(() => {
    const element = ref.current
    if (!enabled || !element) {
      setFits(true)
      return
    }
    const names = key.split("\u0000")
    let active = true
    const check = () => {
      if (!active) return
      const available = element.clientWidth
      if (available === 0) return // hidden (collapsed section): keep the last answer
      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16
      // the segments' text: 0.8rem, medium weight, the page font
      const needed = segmentedWidth(names, `500 ${0.8 * rem}px ${getComputedStyle(element).fontFamily}`)
      setFits(needed === null || needed <= available)
    }
    check()
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(check)
    observer?.observe(element)
    void document.fonts?.ready.then(check)
    return () => {
      active = false
      observer?.disconnect()
    }
  }, [ref, key, enabled])
  return fits
}
