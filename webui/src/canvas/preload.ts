/** Starts fetching the editor chunk (e.g. when the img2img page opens), so the editor shows at once. */
export const preloadImageEditor = () => import("./ui/Editor")
