import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { createFormStore } from "@/ui/form-store"
import type { CustomNode } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import TaskRunner from "./index"
import zh from "./locales/zh-CN.json"
import source from "./index.tsx?raw"

const OWNER = "page:caption"
const NODE: CustomNode = { type: "custom", name: "run", label: "", widget: "task-runner", props: {}, default: null }
const PROPS = { action: "caption", label: "Caption image", interval: 120 }

type Answer = Record<string, unknown>
const json = (body: unknown) => new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } })
const status = (state: string, extra: Answer = {}) => ({ id: "t1", name: "caption", state, progress: null, text: null, error: null, done: ["completed", "failed", "cancelled"].includes(state), ...extra })

/** Answers per action from queues; records every request. */
function server(answers: Record<string, Answer[]>) {
  const calls: { action: string; body: Answer }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    const action = new URL(request.url).pathname.split("/").at(-1)!
    calls.push({ action, body: (await request.json()) as Answer })
    const queue = answers[action] ?? []
    return json(queue.length > 1 ? queue.shift() : (queue[0] ?? {}))
  })
  return calls
}

function setup(formId = `page:caption-${Math.random()}`) {
  const store = createFormStore(formId, { persistDrafts: false })
  store.getState().load([{ owner: OWNER, layout: null, defaults: { image: "img-1" } }])
  render(
    <FormProvider formId={formId} store={store}>
      <OwnerProvider owner={OWNER}>
        <TaskRunner value={null} onChange={() => {}} props={PROPS} node={NODE} disabled={false} id="run" label="" />
      </OwnerProvider>
    </FormProvider>,
  )
  return store
}

describe("task-runner widget", () => {
  afterEach(() => vi.restoreAllMocks())

  it("starts the task, shows its progress and applies the result once it ends", async () => {
    const calls = server({
      caption: [{ __outputs__: { "task:caption": status("queued") } }],
      __task__: [
        { __outputs__: { "task:caption": null } }, // the mount's resume check
        { __outputs__: { "task:caption": status("running", { progress: 0.5, text: { text: "{{index}}/{{total}}", params: { index: 1, total: 2 } } }) } },
        { caption_out: { value: "a cat" }, __outputs__: { "task:caption": status("completed", { progress: 1 }), result: "done" } },
      ],
    })
    const store = setup()
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(calls[0]).toMatchObject({ action: "__task__", body: { task: "caption", resume: true } })

    fireEvent.click(screen.getByRole("button", { name: "Caption image" }))
    await waitFor(() => {
      expect(screen.getByText("1/2")).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Caption image" })).toBeDisabled()
      expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50")
      // the live region announces the state, not every progress tick
      expect(screen.getByRole("status")).toHaveTextContent(/^Caption image: Running…$/)
    })
    await waitFor(() => expect(store.getState().values[OWNER]?.caption_out).toBe("a cat"))
    expect(store.getState().outputs[OWNER]?.result).toBe("done")
    expect(calls.slice(1).map((call) => call.action)).toEqual(["caption", "__task__", "__task__"])
    expect(calls[1]!.body).toMatchObject({ values: { image: "img-1" } })
    expect(calls[2]!.body).toMatchObject({ task: "caption", id: "t1" })
    await waitFor(() => expect(screen.getByRole("button", { name: "Caption image" })).toBeEnabled())
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Caption image: Finished"))
  })

  it("applies the result of a task that ended while no view followed it (a reload)", async () => {
    const calls = server({
      __task__: [{ caption_out: { value: "late cat" }, __outputs__: { "task:caption": status("completed", { progress: 1 }) } }],
    })
    const store = setup()
    await waitFor(() => expect(store.getState().values[OWNER]?.caption_out).toBe("late cat"))
    expect(calls).toHaveLength(1)
    expect(calls[0]).toMatchObject({ action: "__task__", body: { task: "caption", resume: true } })
  })

  it("follows a task that already runs when it mounts, and cancels it", async () => {
    const calls = server({
      __task__: [
        { __outputs__: { "task:caption": status("running") } },
        { __outputs__: { "task:caption": status("running", { cancelling: true }) } },
        { __outputs__: { "task:caption": status("cancelled") }, __toast__: { level: "info", message: "Cancelled" } },
      ],
      __cancel__: [{ __outputs__: { "task:caption": status("running", { cancelling: true }) } }],
    })
    setup()
    const cancel = await screen.findByRole("button", { name: "Cancel" })
    expect(screen.getByText("Running…")).toBeInTheDocument()
    fireEvent.click(cancel)
    await waitFor(() => expect(calls.some((call) => call.action === "__cancel__" && call.body.task === "caption")).toBe(true))
    await waitFor(() => expect(screen.queryByRole("button", { name: /Cancel/ })).not.toBeInTheDocument())
  })

  it("shows why a task failed", async () => {
    server({
      caption: [{ __outputs__: { "task:caption": status("queued") } }],
      __task__: [{ __outputs__: { "task:caption": null } }, { __outputs__: { "task:caption": status("failed", { error: "out of memory" }) } }],
    })
    setup()
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Caption image" }))
    })
    expect(await screen.findByRole("alert")).toHaveTextContent("Failed: out of memory")
  })

  it("translates every chrome string", () => {
    const catalog = zh as Record<string, string>
    expect(widgetKeys(source).filter((key) => !catalog[key])).toEqual([])
  })
})
