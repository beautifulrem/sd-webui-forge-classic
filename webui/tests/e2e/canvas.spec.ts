/** The img2img image editor against the mock backend. */
import { expect, type Locator, type Page, test } from "@playwright/test"

const mod = process.platform === "darwin" ? "Meta" : "Control"

/** A PNG made in the page (a gradient, so the image is visibly there). */
async function makePng(page: Page, width: number, height: number): Promise<Buffer> {
  const data = await page.evaluate(
    ([w, h]) => {
      const canvas = document.createElement("canvas")
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext("2d")!
      const gradient = ctx.createLinearGradient(0, 0, w, h)
      gradient.addColorStop(0, "#2a6df4")
      gradient.addColorStop(1, "#f4b42a")
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, w, h)
      return canvas.toDataURL("image/png").split(",")[1]!
    },
    [width, height] as const,
  )
  return Buffer.from(data, "base64")
}

/** An opaque black PNG with a white rectangle over its left half (an uploaded mask: white = masked). */
async function makeMask(page: Page, width: number, height: number): Promise<Buffer> {
  const data = await page.evaluate(
    ([w, h]) => {
      const canvas = document.createElement("canvas")
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext("2d")!
      ctx.fillStyle = "#000"
      ctx.fillRect(0, 0, w, h)
      ctx.fillStyle = "#fff"
      ctx.fillRect(0, 0, w / 2, h)
      return canvas.toDataURL("image/png").split(",")[1]!
    },
    [width, height] as const,
  )
  return Buffer.from(data, "base64")
}

type MockModule = { handleMockRequest: (request: Request) => Promise<Response> }

/** The core params of the newest img2img job as the mock backend received them (`POST /jobs`). */
async function lastJobCore(page: Page): Promise<Record<string, unknown>> {
  return page.evaluate(async () => {
    // The app's own instance of the in-browser mock server: the exact module URL it loaded (with its version query).
    const loaded = performance.getEntriesByType("resource").map((entry) => entry.name)
    const path = loaded.find((name) => new URL(name).pathname.endsWith("/src/api/mock/server.ts"))
    if (!path) throw new Error(`mock server module not loaded: ${loaded.filter((name) => name.includes("mock")).join(", ")}`)
    const { handleMockRequest } = (await import(/* @vite-ignore */ path)) as MockModule
    const get = async (url: string) => (await handleMockRequest(new Request(`${location.origin}/api/v2${url}`))).json()
    const list = (await get("/jobs?tab=img2img&limit=1")) as { items: { id: string }[] }
    const job = (await get(`/jobs/${list.items[0]!.id}`)) as { core: Record<string, unknown> }
    return job.core
  })
}

/** A file in a DataTransfer made in the page (drop and paste payloads). */
async function transferOf(page: Page, png: Buffer, name: string) {
  return page.evaluateHandle(
    ([b64, fileName]) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
      const transfer = new DataTransfer()
      transfer.items.add(new File([bytes], fileName, { type: "image/png" }))
      return transfer
    },
    [png.toString("base64"), name] as const,
  )
}

const mode = (page: Page, name: string) => page.getByRole("radiogroup", { name: "Mode" }).getByRole("radio", { name, exact: true })

async function stroke(page: Page, from: { x: number; y: number }, to: { x: number; y: number }) {
  await page.mouse.move(from.x, from.y)
  await page.mouse.down()
  for (let i = 1; i <= 10; i++) await page.mouse.move(from.x + ((to.x - from.x) * i) / 10, from.y + ((to.y - from.y) * i) / 10)
  await page.mouse.up()
}

/** Alpha of the paint layer's canvas under a page point (the second Konva canvas). */
async function paintAlpha(surface: Locator, x: number, y: number): Promise<number> {
  return surface.evaluate(
    (host, [px, py]) => {
      const canvases = host.querySelectorAll("canvas")
      const paint = canvases[1] as HTMLCanvasElement
      const rect = paint.getBoundingClientRect()
      const ratio = paint.width / rect.width
      const pixel = paint.getContext("2d")!.getImageData(Math.round((px - rect.left) * ratio), Math.round((py - rect.top) * ratio), 1, 1).data
      return pixel[3]!
    },
    [x, y] as const,
  )
}

async function openWithImage(page: Page) {
  await page.goto("/#/img2img")
  // The first visit compiles the page on the dev server.
  await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible({ timeout: 90_000 })
  const png = await makePng(page, 640, 480)
  const chooser = page.waitForEvent("filechooser")
  await page.getByRole("button", { name: "Add a source image" }).click()
  await (await chooser).setFiles({ name: "source.png", mimeType: "image/png", buffer: png })
  const surface = page.getByRole("img", { name: /^Image editor, 640×480/ })
  await expect(surface).toBeVisible()
  return surface
}

test.describe("image editor", () => {
  test.describe.configure({ timeout: 150_000 })

  test("paints a mask, undoes and redoes it, keeps it across the stage views", async ({ page }) => {
    const surface = await openWithImage(page)
    // img2img mode: nothing to paint, no tools.
    await expect(page.getByRole("toolbar", { name: "Tools" })).toHaveCount(0)
    await page.getByRole("radiogroup", { name: "Mode" }).getByRole("radio", { name: "Inpaint", exact: true }).click()
    await expect(page.getByRole("toolbar", { name: "Tools" })).toBeVisible()
    await expect(page.getByRole("button", { name: "Brush", exact: true })).toHaveAttribute("aria-pressed", "true")

    const box = (await surface.boundingBox())!
    const from = { x: box.x + box.width * 0.35, y: box.y + box.height * 0.5 }
    const to = { x: box.x + box.width * 0.65, y: box.y + box.height * 0.5 }
    await page.mouse.move(from.x, from.y)
    await page.mouse.down()
    for (let i = 1; i <= 12; i++) await page.mouse.move(from.x + ((to.x - from.x) * i) / 12, from.y)
    await page.mouse.up()
    const mid = { x: (from.x + to.x) / 2, y: from.y }
    await expect.poll(() => paintAlpha(surface, mid.x, mid.y)).toBeGreaterThan(0)
    // The layer is uploaded once the user pauses.
    await expect(page.getByRole("status").filter({ hasText: "Saving" })).toHaveCount(0, { timeout: 5_000 })

    await page.keyboard.press(`${mod}+Z`)
    await expect.poll(() => paintAlpha(surface, mid.x, mid.y)).toBe(0)
    await page.keyboard.press(`${mod}+Y`)
    await expect.poll(() => paintAlpha(surface, mid.x, mid.y)).toBeGreaterThan(0)

    // Generating shows the result; the job carries the mask on screen; the source still has the mask.
    await page.getByRole("button", { name: /^Generate/ }).first().click()
    await expect(page.getByRole("tab", { name: "Result" })).toHaveAttribute("aria-selected", "true")
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    const core = await lastJobCore(page)
    expect(core.mode).toBe("Inpaint")
    expect(core.init_img).toMatch(/^upload-/)
    expect(core.mask).toMatch(/^upload-/)
    expect(core.mask).not.toBe(core.init_img)
    await page.getByRole("tab", { name: "Source" }).click()
    const again = page.getByRole("img", { name: /^Image editor, 640×480/ })
    await expect(again).toBeVisible()
    await expect.poll(() => paintAlpha(again, mid.x, mid.y)).toBeGreaterThan(0)
  })

  test("sketches in colour, picks colours and erases with Shift", async ({ page }) => {
    const surface = await openWithImage(page)
    await page.getByRole("radiogroup", { name: "Mode" }).getByRole("radio", { name: "Sketch", exact: true }).click()
    await expect(page.getByRole("button", { name: "Eyedropper" })).toBeVisible()
    await expect(page.getByRole("button", { name: "Brush color" })).toBeVisible()
    await page.getByRole("slider", { name: "Opacity" }).first().focus()

    const box = (await surface.boundingBox())!
    const y = box.y + box.height * 0.3
    const stroke = async (x0: number, x1: number) => {
      await page.mouse.move(x0, y)
      await page.mouse.down()
      for (let i = 1; i <= 8; i++) await page.mouse.move(x0 + ((x1 - x0) * i) / 8, y)
      await page.mouse.up()
    }
    await stroke(box.x + box.width * 0.3, box.x + box.width * 0.7)
    const mid = box.x + box.width * 0.5
    await expect.poll(() => paintAlpha(surface, mid, y)).toBeGreaterThan(200)
    // Shift turns the brush into the eraser.
    await page.keyboard.down("Shift")
    await stroke(box.x + box.width * 0.45, box.x + box.width * 0.55)
    await page.keyboard.up("Shift")
    await expect.poll(() => paintAlpha(surface, mid, y)).toBe(0)
    await expect.poll(() => paintAlpha(surface, box.x + box.width * 0.35, y)).toBeGreaterThan(200)
  })

  test("zooms, fits, maximises and lists its shortcuts", async ({ page }) => {
    const surface = await openWithImage(page)
    const zoom = page.getByRole("button", { name: /^Zoom \d+%/ })
    const before = await zoom.textContent()
    await surface.hover()
    await page.mouse.wheel(0, -400)
    await expect(zoom).not.toHaveText(before!)
    await page.keyboard.press("r")
    await expect(zoom).toHaveText(before!)

    await page.keyboard.press("?")
    await expect(page.getByRole("heading", { name: "Keyboard shortcuts" })).toBeVisible()
    await page.keyboard.press("Escape")

    await surface.hover()
    await page.keyboard.press("f")
    const dialog = page.getByRole("dialog", { name: "Source image" })
    await expect(dialog).toBeVisible()
    await expect(dialog.getByRole("img", { name: /^Image editor, 640×480/ })).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(dialog).toBeHidden()
  })

  test("takes its shortcuts when it appears under a resting pointer (no pointer enter)", async ({ page }) => {
    const surface = await openWithImage(page)
    await mode(page, "Inpaint").click()
    const box = (await surface.boundingBox())!
    const from = { x: box.x + box.width * 0.35, y: box.y + box.height * 0.5 }
    const to = { x: box.x + box.width * 0.65, y: from.y }
    const mid = { x: (from.x + to.x) / 2, y: from.y }
    await stroke(page, from, to)
    await expect.poll(() => paintAlpha(surface, mid.x, mid.y)).toBeGreaterThan(0)
    await expect(page.getByRole("status").filter({ hasText: "Saving" })).toHaveCount(0, { timeout: 5_000 })

    // Switch to Result and back from the keyboard: a new editor mounts under the pointer, which does not move.
    const source = page.getByRole("tab", { name: "Source", exact: true })
    await source.focus()
    await page.keyboard.press("ArrowRight")
    await expect(page.getByRole("tab", { name: "Result", exact: true })).toHaveAttribute("aria-selected", "true")
    await page.keyboard.press("ArrowLeft")
    const again = page.getByRole("img", { name: /^Image editor, 640×480/ })
    await expect(again).toBeVisible()
    await expect.poll(() => paintAlpha(again, mid.x, mid.y)).toBeGreaterThan(0)

    await page.keyboard.press(`${mod}+Z`)
    await expect.poll(() => paintAlpha(again, mid.x, mid.y)).toBe(0)
    await page.keyboard.press("f")
    const dialog = page.getByRole("dialog", { name: "Source image" })
    await expect(dialog).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(dialog).toBeHidden()
  })

  test("shows an uploaded mask on the canvas, keeps each mode's layer and its own undo", async ({ page }) => {
    const surface = await openWithImage(page)
    const box = (await surface.boundingBox())!
    const left = { x: box.x + box.width * 0.3, y: box.y + box.height * 0.5 }
    const right = { x: box.x + box.width * 0.7, y: box.y + box.height * 0.5 }

    // Sketch something first.
    await mode(page, "Sketch").click()
    await stroke(page, { x: right.x - 30, y: box.y + box.height * 0.3 }, { x: right.x + 30, y: box.y + box.height * 0.3 })
    const sketched = { x: right.x, y: box.y + box.height * 0.3 }
    await expect.poll(() => paintAlpha(surface, sketched.x, sketched.y)).toBeGreaterThan(0)

    // Inpaint upload: the mask uploaded in the Source panel shows on the canvas (white = masked).
    await mode(page, "Inpaint upload").click()
    const chooser = page.waitForEvent("filechooser")
    await page.getByRole("button", { name: "Upload Mask" }).click()
    await (await chooser).setFiles({ name: "mask.png", mimeType: "image/png", buffer: await makeMask(page, 640, 480) })
    await expect.poll(() => paintAlpha(surface, left.x, left.y)).toBeGreaterThan(0)
    await expect.poll(() => paintAlpha(surface, right.x, right.y)).toBe(0)
    // Undo here never touches the hidden sketch.
    await surface.hover()
    await page.keyboard.press(`${mod}+Z`)

    await mode(page, "Sketch").click()
    await expect.poll(() => paintAlpha(surface, sketched.x, sketched.y)).toBeGreaterThan(0)
    await expect(page.getByRole("status").filter({ hasText: "Saving" })).toHaveCount(0, { timeout: 5_000 })
    await page.getByRole("button", { name: /^Generate/ }).first().click()
    await expect(page.getByRole("tab", { name: "Result" })).toHaveAttribute("aria-selected", "true")
    const core = await lastJobCore(page)
    expect(core.mode).toBe("Sketch")
    expect(core.sketch).toMatch(/^upload-/)
    expect(core.mask_upload).toMatch(/^upload-/)
  })

  test("takes images dropped on it and pasted while the pointer is over it", async ({ page }) => {
    await page.goto("/#/img2img")
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible({ timeout: 90_000 })
    const editor = page.locator('[data-slot="image-editor"]').first()
    await expect(page.getByRole("button", { name: "Add a source image" })).toBeVisible()
    await editor.dispatchEvent("drop", { dataTransfer: await transferOf(page, await makePng(page, 320, 240), "dropped.png") })
    const dropped = page.getByRole("img", { name: /^Image editor, 320×240/ })
    await expect(dropped).toBeVisible()

    await dropped.hover()
    const pasted = await transferOf(page, await makePng(page, 200, 100), "pasted.png")
    await page.evaluate((clipboardData) => document.body.dispatchEvent(new ClipboardEvent("paste", { clipboardData, bubbles: true, cancelable: true })), pasted)
    await expect(page.getByRole("img", { name: /^Image editor, 200×100/ })).toBeVisible()
  })

  test("takes the image size and removes the image", async ({ page }) => {
    await openWithImage(page)
    await page.getByRole("button", { name: "Use the image size (640×480)" }).first().click()
    await expect(page.getByRole("slider", { name: "Width" }).first()).toHaveAttribute("aria-valuenow", "640")
    await page.getByRole("button", { name: "Remove image" }).click()
    await expect(page.getByRole("button", { name: "Add a source image" })).toBeVisible()
  })

  test("leaves one-finger scrolling to the page while there is nothing to paint @phone", async ({ page }) => {
    const surface = await openWithImage(page)
    const touchAction = () => surface.evaluate((host) => getComputedStyle(host).touchAction)
    // img2img mode: a swipe over the image scrolls the page; two fingers still pan and zoom it.
    expect(await touchAction()).toBe("pan-y")
    await mode(page, "Inpaint").click()
    await expect.poll(touchAction).toBe("none")
  })
})
