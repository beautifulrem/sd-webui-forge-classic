import { afterEach, describe, expect, it, vi } from "vitest"
import { ApiError, http, validationErrors } from "./http"

const reply = (body: unknown, status: number) =>
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }))

describe("http errors", () => {
  afterEach(() => vi.restoreAllMocks())

  it("maps a rejected request's validation errors (schema-format.md §3)", async () => {
    const errors = [
      { owner: "core", param: "width", message: "expected a number, got 'abc'" },
      { owner: "sampler", param: "steps", message: "must be at most 150" },
    ]
    reply({ errors }, 422)
    const error = await http.post("/jobs", {}).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).status).toBe(422)
    expect((error as ApiError).message).toBe("width: expected a number, got 'abc'; sampler.steps: must be at most 150")
    expect(validationErrors(error)).toEqual(errors)
  })

  it("keeps FastAPI details readable and ignores malformed error lists", async () => {
    reply({ detail: [{ msg: "field required", loc: ["body", "tab"] }] }, 422)
    const error = await http.post("/jobs", {}).catch((e: unknown) => e)
    expect((error as ApiError).message).toBe("field required")
    expect(validationErrors(error)).toBeNull()
    expect(validationErrors(new ApiError(400, "x", { errors: [{ owner: 1 }] }))).toBeNull()
  })
})
