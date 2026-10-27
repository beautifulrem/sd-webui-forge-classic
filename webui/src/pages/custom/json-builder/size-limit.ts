import type { FormTab } from "@/api/types"
import { getFormStore } from "@/ui/form-store"

/**
 * The largest side form `tab` accepts (the maximum of its core Width / Height
 * sliders) when `width` × `height` goes past it, else null. txt2img goes up to
 * 4096 and img2img to 2048, while the canvas has no upper bound: a size past
 * the form's limit is only rejected when generating, so the page warns.
 */
export function sizeLimit(tab: FormTab, width: number, height: number): number | null {
  const core = getFormStore(tab).getState().index.inputs.get("core")
  const maximum = (name: string) => {
    const node = core?.get(name)
    return node && (node.type === "slider" || node.type === "number") && typeof node.maximum === "number" ? node.maximum : Number.POSITIVE_INFINITY
  }
  const exceeded = [maximum("width"), maximum("height")].filter((max, i) => [width, height][i]! > max)
  return exceeded.length > 0 ? Math.min(...exceeded) : null
}
