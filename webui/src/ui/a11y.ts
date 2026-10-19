/** ids of a field's help texts, for aria-describedby / aria-labelledby. */
export const labelId = (id: string) => `${id}-label`
export const infoTextId = (id: string) => `${id}-info`
export const errorTextId = (id: string) => `${id}-error`
export const hintTextId = (id: string) => `${id}-tooltip`

/** The id of a field's hint (tooltip) text when it has one: an `extra` id for `describedBy`. */
export const hintRef = (id: string, hint?: string | null) => (hint ? hintTextId(id) : undefined)

/** aria-describedby of a field's control: its info and error texts, plus `extra` ids. */
export function describedBy(id: string, info?: string | null, error?: string | null, ...extra: (string | undefined)[]) {
  const ids = [...extra, info ? infoTextId(id) : undefined, error ? errorTextId(id) : undefined].filter(Boolean)
  return ids.length > 0 ? ids.join(" ") : undefined
}
