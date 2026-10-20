/**
 * Image editor module (Konva is imported only by `engine/renderer.ts`, inside
 * the lazily loaded editor chunk). Public surface:
 *
 * - `ImageEditor` (+ `preloadImageEditor`) and its props (`types.ts`);
 * - the img2img integration: `Img2imgSourcePanel` (parameters pane) and
 *   `Img2imgStage` / `useImg2imgStageView` (the generate stage); the two
 *   components load with their first render (`img2img/lazy.tsx`).
 */
export { ImageEditor } from "./ImageEditor"
export { preloadImageEditor } from "./preload"
export type { EditorLayer, ImageEditorProps, LayerKind } from "./types"
export { editorBinding, modeLayers, type EditorBinding, type LayerBinding, type ModeLayers } from "./core/modes"
// lazy: only the img2img tab renders them (txt2img's first load stays without them)
export { Img2imgSourcePanel, Img2imgStage } from "./img2img/lazy"
export { useImg2imgStageView, type StageView } from "./img2img/stage-view"
