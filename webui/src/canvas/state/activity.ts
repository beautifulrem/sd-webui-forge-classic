/**
 * Painting gestures in progress in any editor (a stroke, a shape, a pick).
 * Tiny and Konva-free, so the main bundle can ask: the img2img stage never
 * switches view under the user's brush.
 */
const painting = new Set<object>()

/** Records whether `owner` (an editor's input handler) has a painting gesture in progress. */
export function setPainting(owner: object, on: boolean) {
  if (on) painting.add(owner)
  else painting.delete(owner)
}

/** Whether the user is painting in some editor right now. */
export const isPainting = () => painting.size > 0
