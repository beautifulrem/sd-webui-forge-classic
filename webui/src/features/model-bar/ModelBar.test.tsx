import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { bootstrap, models } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import ModelBar, { MODULES_HELP_URL } from "./ModelBar"

function renderBar() {
  return render(
    <QueryClientProvider client={queryClient}>
      <ModelBar />
    </QueryClientProvider>,
  )
}

/** The picker's own trigger (the narrow-header popover holds a second one once open). */
const modulesTriggers = () => screen.getAllByRole("button", { name: "VAE / Text encoders" })

describe("model bar modules picker", () => {
  let patch: ReturnType<typeof vi.spyOn>
  beforeEach(() => {
    queryClient.setQueryData(keys.app, structuredClone(bootstrap))
    queryClient.setQueryData(keys.models, models)
    patch = vi.spyOn(api, "patchOptions").mockResolvedValue({} as never)
    vi.spyOn(api, "app").mockResolvedValue(structuredClone(bootstrap))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("applies the selection once, when the picker closes", async () => {
    renderBar()
    fireEvent.click(modulesTriggers()[0]!)
    fireEvent.click(await screen.findByRole("checkbox", { name: "ae.safetensors" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "clip_l.safetensors" }))
    expect(patch).not.toHaveBeenCalled()
    fireEvent.keyDown(screen.getByRole("checkbox", { name: "ae.safetensors" }), { key: "Escape" })
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1))
    expect(patch).toHaveBeenCalledWith({ forge_additional_modules: ["ae.safetensors", "clip_l.safetensors"] })
  })

  it("keeps a selection made in the narrow header's popover when everything closes at once", async () => {
    const { unmount } = renderBar()
    fireEvent.click(screen.getByRole("button", { name: "Model settings" }))
    // The nested picker inside the settings popover.
    await waitFor(() => expect(modulesTriggers()).toHaveLength(2))
    fireEvent.click(modulesTriggers()[1]!)
    fireEvent.click(await screen.findByRole("checkbox", { name: "qwen_image_vae.safetensors" }))
    // The page goes away (or the outer popover unmounts its content) without the picker closing itself.
    unmount()
    expect(patch).toHaveBeenCalledTimes(1)
    expect(patch).toHaveBeenCalledWith({ forge_additional_modules: ["qwen_image_vae.safetensors"] })
  })

  it("writes nothing when the selection did not change", async () => {
    const { unmount } = renderBar()
    fireEvent.click(modulesTriggers()[0]!)
    const box = await screen.findByRole("checkbox", { name: "ae.safetensors" })
    fireEvent.click(box)
    fireEvent.click(box)
    unmount()
    expect(patch).not.toHaveBeenCalled()
  })

  it("says what the modules are for and links to where to get them", async () => {
    renderBar()
    fireEvent.click(modulesTriggers()[0]!)
    const link = await screen.findByRole("link", { name: "Where to get them" })
    expect(link).toHaveAttribute("href", MODULES_HELP_URL)
    expect(link).toHaveAttribute("target", "_blank")
    expect(link.getAttribute("rel")).toContain("noopener")
    expect(screen.getByText(/Checkpoints without a built-in VAE or text encoder/)).toBeVisible()
  })
})
