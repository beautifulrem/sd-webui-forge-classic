/**
 * Extras against the mock backend: upload a source, pick the upscaler and the scale, process,
 * then the result (its size, the before / after comparison, the viewer) and its infotext; a
 * generated image sent to Extras; Hires. fix and Refiner values in a txt2img job and its infotext.
 */
import { expect, type Page, test } from "@playwright/test"
import { mod, openGenerate, openRoute, setPrompt, watchConsole } from "./helpers"

type MockModule = { handleMockRequest: (request: Request) => Promise<Response> }

/** The newest job of `tab` as the in-page mock server stored it (its named values). */
async function lastJob(page: Page, tab: string): Promise<{ core: Record<string, unknown>; scripts?: Record<string, Record<string, unknown>>; outputs: { id: string; width: number; height: number; infotext?: string }[] }> {
  return page.evaluate(async (jobTab) => {
    // The app's own instance of the in-browser mock server: the exact module URL it loaded.
    const loaded = performance.getEntriesByType("resource").map((entry) => entry.name)
    const path = loaded.find((name) => new URL(name).pathname.endsWith("/src/api/mock/server.ts"))
    if (!path) throw new Error("mock server module not loaded")
    const { handleMockRequest } = (await import(/* @vite-ignore */ path)) as MockModule
    const get = async (url: string) => (await handleMockRequest(new Request(`${location.origin}/api/v2${url}`))).json()
    const list = (await get(`/jobs?tab=${jobTab}&limit=1`)) as { items: { id: string }[] }
    return get(`/jobs/${list.items[0]!.id}`)
  }, tab)
}

/** A PNG of the given size, drawn in the page. */
async function makePng(page: Page, width: number, height: number): Promise<Buffer> {
  const data = await page.evaluate(
    ([w, h]) => {
      const canvas = document.createElement("canvas")
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext("2d")!
      ctx.fillStyle = "#16a34a"
      ctx.fillRect(0, 0, w, h)
      ctx.fillStyle = "#f59e0b"
      ctx.fillRect(w / 4, h / 4, w / 2, h / 2)
      return canvas.toDataURL("image/png").split(",")[1]!
    },
    [width, height] as const,
  )
  return Buffer.from(data, "base64")
}

test.describe("extras", () => {
  test("upscales an uploaded image with the chosen upscaler and scale", async ({ page }) => {
    const console = watchConsole(page)
    await openRoute(page, "extras")
    const form = page.getByRole("form", { name: "Parameters" })
    await expect(page.getByText("Results appear here, next to their source.")).toBeVisible()

    // Upload the source
    const png = await makePng(page, 300, 200)
    const chooser = page.waitForEvent("filechooser")
    await form.getByRole("button", { name: "Source" }).click()
    await (await chooser).setFiles({ name: "photo.png", mimeType: "image/png", buffer: png })
    await expect(form.getByRole("button", { name: "Remove Source" })).toBeVisible()

    // Upscaler and scale
    await form.getByRole("combobox", { name: "Upscaler" }).click()
    await page.getByRole("option", { name: "R-ESRGAN 4x+" }).click()
    await expect(form.getByRole("combobox", { name: "Upscaler" })).toHaveText(/R-ESRGAN 4x\+/)
    const scale = form.getByRole("textbox", { name: "Scale by" })
    await scale.fill("3")
    await scale.press("Tab")
    await expect(form.getByRole("slider", { name: "Scale by" })).toHaveAttribute("aria-valuenow", "3")

    // Process: progress, then the result next to its source
    await page.getByRole("button", { name: /^Process/ }).click()
    await expect(page.getByRole("button", { name: "Interrupt" })).toBeVisible()
    await expect(page.getByRole("button", { name: "Interrupt" })).toBeHidden({ timeout: 15_000 })
    const result = page.getByRole("region", { name: "Result" })
    await expect(result.getByRole("slider", { name: "Comparison divider" })).toBeVisible()
    await expect(result.getByText("900×600")).toBeVisible()
    await expect(result.getByText("Postprocess upscale by: 3, Postprocess upscaler: R-ESRGAN 4x+")).toBeVisible()

    // The job carried the form's values by name
    const job = await lastJob(page, "extras")
    expect(job.core).toMatchObject({ upscaler_1: "R-ESRGAN 4x+", upscaling_resize: 3 })
    expect(job.core.image).toMatch(/^upload-/)
    expect(job.outputs).toHaveLength(1)
    expect(job.outputs[0]).toMatchObject({ width: 900, height: 600, infotext: "Postprocess upscale by: 3, Postprocess upscaler: R-ESRGAN 4x+" })

    // Result only, then the viewer
    await result.getByRole("radio", { name: "Result" }).click()
    await expect(result.getByRole("img", { name: "Processed image" })).toHaveAttribute("src", new RegExp(`/files/${job.outputs[0]!.id}`))
    await result.getByRole("button", { name: "Open in the viewer" }).first().click()
    const viewer = page.getByRole("dialog")
    await expect(viewer).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(viewer).toBeHidden()
    console.assertClean()
  })

  test("processes an image sent from a txt2img result", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await setPrompt(page, "a lighthouse at dusk")
    await page.keyboard.press(`${mod}+Enter`)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    const generated = (await lastJob(page, "txt2img")).outputs[0]!
    await page.getByRole("button", { name: "Send to" }).click()
    await page.getByRole("menuitem", { name: "Extras", exact: true }).click()
    await expect(page).toHaveURL(/#\/extras$/)

    const form = page.getByRole("form", { name: "Parameters" })
    await expect(form.getByRole("button", { name: "Remove Source" })).toBeVisible()
    await page.keyboard.press(`${mod}+Enter`)
    const result = page.getByRole("region", { name: "Result" })
    await expect(result.getByText("Postprocess upscale by: 2, Postprocess upscaler: Lanczos")).toBeVisible({ timeout: 15_000 })
    const job = await lastJob(page, "extras")
    expect(job.core.image).toBe(generated.id) // the output itself, no new upload
    expect(job.outputs[0]).toMatchObject({ width: generated.width * 2, height: generated.height * 2 })
    await expect(result.getByText(`${generated.width * 2}×${generated.height * 2}`)).toBeVisible()
    console.assertClean()
  })
})

test.describe("hires fix and refiner", () => {
  test("generates with Hires. fix and its values reach the job and the infotext", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await setPrompt(page, "a castle on a hill")
    // The section header carries the enable switch; switching it on opens the section.
    const toggle = page.getByRole("switch", { name: "Enable Hires. fix", exact: true })
    await toggle.click()
    await expect(toggle).toBeChecked()
    const section = page.getByRole("region", { name: "Hires. fix", exact: true })
    const upscaleBy = section.getByRole("textbox", { name: "Upscale by" })
    await upscaleBy.fill("1.5")
    await upscaleBy.press("Tab")
    await section.getByRole("combobox", { name: "Upscaler" }).click()
    await page.getByRole("option", { name: "R-ESRGAN 4x+" }).click()

    await page.keyboard.press(`${mod}+Enter`)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    const job = await lastJob(page, "txt2img")
    expect(job.core).toMatchObject({ enable_hr: true, hr_scale: 1.5, hr_upscaler: "R-ESRGAN 4x+" })
    const width = Number(job.core.width)
    const height = Number(job.core.height)
    expect(job.outputs[0]).toMatchObject({ width: Math.round(width * 1.5), height: Math.round(height * 1.5) })
    expect(job.outputs[0]!.infotext).toContain("Hires upscale: 1.5, Hires upscaler: R-ESRGAN 4x+")
    // The result's parameters list shows the Hires. fix values from the infotext
    const result = page.getByRole("region", { name: "Result" })
    const value = (term: string) => result.getByRole("term").filter({ hasText: new RegExp(`^${term}$`) }).locator("xpath=following-sibling::*[1]")
    await expect(value("Hires upscale")).toHaveText("1.5")
    await expect(value("Hires upscaler")).toHaveText("R-ESRGAN 4x+")
    console.assertClean()
  })

  test("Hires. fix dims the target controls that do not apply, and a target size wins over the factor", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await setPrompt(page, "a castle on a hill")
    await page.getByRole("switch", { name: "Enable Hires. fix", exact: true }).click()
    const section = page.getByRole("region", { name: "Hires. fix", exact: true })
    const field = (name: string) => section.getByRole("textbox", { name }).locator("xpath=ancestor::div[contains(@class, 'group/field')][1]")
    const inactive = async (name: string) => (await field(name).getAttribute("data-inactive")) === "true"
    const states = async () => [await inactive("Upscale by"), await inactive("Resize width to"), await inactive("Resize height to")]
    expect(await states()).toEqual([false, true, true])

    // a dimmed control stays editable: typing a width makes it the one in use
    const resizeWidth = section.getByRole("textbox", { name: "Resize width to" })
    await expect(resizeWidth).toBeEditable()
    await resizeWidth.fill("1536")
    await resizeWidth.press("Tab")
    await expect.poll(states).toEqual([true, false, true])
    await expect(section.getByText(/to 1536×1536/)).toBeVisible()
    await expect.poll(() => field("Upscale by").evaluate((element) => getComputedStyle(element).opacity)).toBe("0.5")

    await page.keyboard.press(`${mod}+Enter`)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    const job = await lastJob(page, "txt2img")
    expect(job.core).toMatchObject({ enable_hr: true, hr_resize_x: 1536, hr_resize_y: 0 })
    expect(job.outputs[0]).toMatchObject({ width: 1536, height: 1536 })
    expect(job.outputs[0]!.infotext).toContain("Hires resize: 1536x1536")
    console.assertClean()
  })

  test("generates with the Refiner and its checkpoint and switch point reach the job and the infotext", async ({ page }) => {
    const console = watchConsole(page)
    await openGenerate(page)
    await setPrompt(page, "a lighthouse at dusk")
    const toggle = page.getByRole("switch", { name: "Enable Refiner", exact: true })
    await toggle.click()
    await expect(toggle).toBeChecked()
    const section = page.getByRole("region", { name: "Refiner", exact: true })
    await section.getByRole("combobox", { name: "Checkpoint" }).click()
    await page.getByRole("option", { name: "noobai-xl-vpred-1.0.safetensors" }).click()
    const switchAt = section.getByRole("textbox", { name: "Switch at" })
    await switchAt.fill("0.6")
    await switchAt.press("Tab")

    await page.keyboard.press(`${mod}+Enter`)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    const job = await lastJob(page, "txt2img")
    expect(job.scripts?.refiner).toMatchObject({ enable: true, checkpoint: "noobai-xl-vpred-1.0.safetensors", switch_at: 0.6 })
    expect(job.outputs[0]!.infotext).toContain("Refiner: noobai-xl-vpred-1.0.safetensors, Refiner switch at: 0.6")
    const result = page.getByRole("region", { name: "Result" })
    const value = (term: string) => result.getByRole("term").filter({ hasText: new RegExp(`^${term}$`) }).locator("xpath=following-sibling::*[1]")
    await expect(value("Refiner")).toHaveText("noobai-xl-vpred-1.0.safetensors")
    await expect(value("Refiner switch at")).toHaveText("0.6")
    console.assertClean()
  })
})
