/** OpenPose JSON text of the pose editor. */

export type ParsedPose = { ok: true; pose: unknown } | { ok: false; error: string }

/** The pose as indented JSON ("" for none). */
export function poseText(pose: unknown): string {
  return pose === null || pose === undefined ? "" : JSON.stringify(pose, null, 2)
}

/** A pose from editor text: an OpenPose JSON object (or a list of them, the first is used by the server). */
export function parsePose(text: string): ParsedPose {
  let pose: unknown
  try {
    pose = JSON.parse(text)
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
  const first = Array.isArray(pose) ? pose[0] : pose
  if (typeof first !== "object" || first === null || Array.isArray(first)) return { ok: false, error: "expected a JSON object" }
  const record = first as Record<string, unknown>
  for (const key of ["canvas_width", "canvas_height"]) {
    if (typeof record[key] !== "number" || !Number.isFinite(record[key]) || (record[key] as number) < 1) return { ok: false, error: `${key} is missing` }
  }
  return { ok: true, pose }
}
