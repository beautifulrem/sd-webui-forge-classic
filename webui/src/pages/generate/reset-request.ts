/** "Reset all parameters" requests: reset at once, or ask first (`ResetFormDialog`). */
import { create } from "zustand"
import type { GenerateTab } from "@/features/progress/run-controller"
import { resetForm, resetLosesWork } from "./run-helpers"

/** The form waiting for the user's confirmation, if any. */
export const useResetRequest = create<{ tab: GenerateTab | null }>()(() => ({ tab: null }))

/** Resets the form now, or asks first when prompts or images would be lost. */
export function requestReset(tab: GenerateTab) {
  if (resetLosesWork(tab)) useResetRequest.setState({ tab })
  else resetForm(tab)
}
