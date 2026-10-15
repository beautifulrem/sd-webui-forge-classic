import { create } from "zustand"

type SlotState = { element: HTMLElement | null; setElement: (element: HTMLElement | null) => void }

/** The header element pages portal into (see header-slot.tsx). */
export const useHeaderSlotTarget = create<SlotState>()((set) => ({
  element: null,
  setElement: (element) => set({ element }),
}))
