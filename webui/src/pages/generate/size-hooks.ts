/** Hooks that write the core image size as a user edit. */
import { runChangeEffects } from "@/ui/actions"
import { useForm } from "@/ui/hooks"
import type { SliderNode } from "@/ui/schema"
import { type Dimensions, parseSize } from "./size"

/** Sets core width/height as a user edit (clamped to the sliders; runs their on_change). */
export function useSetSize() {
  const { store, formId } = useForm()
  return (size: Dimensions) => {
    const state = store.getState()
    const nodes = state.index.inputs.get("core")
    const clampTo = (name: "width" | "height", value: number) => {
      const node = nodes?.get(name) as SliderNode | undefined
      const override = state.overrides.core?.[name]
      const min = override?.minimum ?? node?.minimum ?? 64
      const max = override?.maximum ?? node?.maximum ?? 4096
      return Math.min(max, Math.max(min, Math.round(value)))
    }
    if (!nodes?.has("width") || !nodes.has("height")) return
    state.setValues("core", { width: clampTo("width", size.width), height: clampTo("height", size.height) })
    runChangeEffects({ store, tab: formId }, "core", ["width", "height"])
  }
}

/** Handles pasted "W x H" text; returns true when it was a size. */
export function usePasteSize() {
  const setSize = useSetSize()
  return (event: React.ClipboardEvent) => {
    const target = event.target as HTMLElement
    // Pasting into a text field that is not a size slider's number box stays a paste.
    if (target.closest("[data-slot=prompt-editor], textarea")) return
    const size = parseSize(event.clipboardData.getData("text"))
    if (!size) return
    event.preventDefault()
    setSize(size)
  }
}
