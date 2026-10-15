import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import openapiTS, { astToString } from "openapi-typescript"
import { describe, expect, it } from "vitest"

const here = (name: string) => fileURLToPath(new URL(name, import.meta.url))

describe("generated API types", () => {
  it("match src/api/openapi.json (run `pnpm gen:api` after tools/export_openapi.py)", async () => {
    const schema = JSON.parse(readFileSync(here("../openapi.json"), "utf8"))
    const generated = astToString(await openapiTS(schema, { emptyObjectsUnknown: true }))
    const committed = readFileSync(here("./schema.d.ts"), "utf8")
    expect(generated).toContain("export interface paths")
    // The CLI prepends a banner comment; compare the declarations.
    expect(committed.endsWith(generated)).toBe(true)
  })
})
