/**
 * Picks the longest placeholder that fits its input, so a hint is never cut
 * mid-word on a phone or in a narrow panel (and translations of any length
 * work): `candidates` go from most to least descriptive; the last one is used
 * when none fits.
 */
import { type RefObject, useLayoutEffect, useState } from "react"

let canvas: HTMLCanvasElement | null = null

function textWidth(text: string, input: HTMLElement): number {
  canvas ??= document.createElement("canvas")
  const context = canvas.getContext("2d")
  if (!context) return 0
  const style = getComputedStyle(input)
  context.font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`
  return context.measureText(text).width
}

/** The room for text inside an input: its width without padding (icons and buttons sit in the padding). */
function room(input: HTMLElement): number {
  const style = getComputedStyle(input)
  return input.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
}

export function pickPlaceholder(candidates: readonly string[], fits: (text: string) => boolean): string {
  return candidates.find(fits) ?? candidates.at(-1) ?? ""
}

export function useFittingPlaceholder(ref: RefObject<HTMLInputElement | null>, candidates: readonly string[]): string {
  const key = candidates.join("\u0000")
  const [chosen, setChosen] = useState(candidates[0] ?? "")
  useLayoutEffect(() => {
    const input = ref.current
    const list = key.split("\u0000")
    if (!input) return
    const update = () => {
      const width = room(input)
      // not laid out yet (hidden tab, jsdom): keep the full text
      if (width <= 0) return setChosen(list[0] ?? "")
      setChosen(pickPlaceholder(list, (text) => textWidth(text, input) <= width))
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(input)
    return () => observer.disconnect()
  }, [ref, key])
  return candidates.includes(chosen) ? chosen : (candidates[0] ?? "")
}
