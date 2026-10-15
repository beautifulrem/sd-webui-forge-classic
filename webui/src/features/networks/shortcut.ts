/** The networks panel shortcut and DOM id (shared by the header button and the lazy drawer). */
export const NETWORKS_SHORTCUT = "Alt+N"

/** Alt+N (by key code: on macOS Option+N is a dead key). */
export const isNetworksShortcut = (event: KeyboardEvent) => event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && event.code === "KeyN"

/** The drawer element (`aria-controls` of the header button). */
export const DRAWER_ID = "remi-networks-drawer"
