/**
 * Layout and feedback checks that need a real browser: the header at
 * 1280×720 (idle and while a job runs), the Generate panes' hide / show
 * toggles (real pane sizes), tab strips on a phone, platform
 * shortcut hints and generation warnings.
 */
import { expect, type Page, test } from "@playwright/test"
import { mod, openGenerate, presetStorage, setPrompt } from "./helpers"

/** Pairs of header controls whose boxes overlap, and kbd hints cut by their button. */
async function headerProblems(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const header = document.querySelector("header")!
    const boxes = [...header.querySelectorAll<HTMLElement>("button, [role=combobox], h1")]
      .map((element) => ({ element, rect: element.getBoundingClientRect(), name: (element.getAttribute("aria-label") || element.textContent || "").trim() }))
      .filter(({ rect }) => rect.width > 0)
    const problems: string[] = []
    for (const [i, a] of boxes.entries()) {
      for (const b of boxes.slice(i + 1)) {
        if (a.element.contains(b.element) || b.element.contains(a.element)) continue
        const x = Math.min(a.rect.right, b.rect.right) - Math.max(a.rect.left, b.rect.left)
        const y = Math.min(a.rect.bottom, b.rect.bottom) - Math.max(a.rect.top, b.rect.top)
        if (x > 0.5 && y > 0.5) problems.push(`"${a.name}" overlaps "${b.name}" by ${x.toFixed(1)}px`)
      }
      if (a.rect.right > innerWidth + 0.5) problems.push(`"${a.name}" leaves the window`)
    }
    for (const kbd of document.querySelectorAll<HTMLElement>("header button kbd, main button kbd")) {
      const k = kbd.getBoundingClientRect()
      const b = kbd.closest("button")!.getBoundingClientRect()
      if (k.width > 0 && (k.right > b.right - 2 || k.left < b.left)) problems.push(`kbd "${kbd.textContent}" is cut by its button`)
    }
    for (const label of document.querySelectorAll<HTMLElement>("main button .truncate")) {
      if (label.scrollWidth > label.clientWidth + 1) problems.push(`button label "${label.textContent}" is cut`)
    }
    return problems
  })
}

test.describe("layout at 1280×720", () => {
  test.use({ viewport: { width: 1280, height: 720 } })

  for (const locale of ["en", "zh-CN"]) {
    test(`the header and the generate bar fit, idle and while a job runs (${locale})`, async ({ page }) => {
      await presetStorage(page, { locale })
      // Quick settings in the header too (the most crowded case).
      await page.goto("/#/settings")
      await page.getByRole("button", { name: /^(Pin .*quick settings|将.*固定)/ }).first().click()
      await expect(page.getByRole("button", { name: /^(Unpin .*quick settings|从快捷设置中取消固定)/ }).first()).toBeVisible()
      await page.evaluate(() => (location.hash = "#/txt2img"))
      await expect(page.getByRole("textbox", { name: /^(Prompt|提示词)$/ })).toBeVisible()
      await expect(page.locator("header").getByRole("button", { name: /^(Quick settings|快捷设置)$/ })).toBeVisible()
      await expect.poll(() => headerProblems(page)).toEqual([])

      await page.getByRole("button", { name: /^(Generate|生成)/ }).first().click()
      await expect(page.getByRole("button", { name: /^(Interrupt|中断)$/ })).toBeVisible()
      expect(await headerProblems(page)).toEqual([])
    })
  }
})

test.describe("generate panes", () => {
  test.use({ viewport: { width: 1440, height: 900 } })

  test("hiding a pane gives its room to the stage and showing it restores the layout", async ({ page }) => {
    await openGenerate(page)
    const sizes = () => page.evaluate(() => ["params", "stage", "history"].map((id) => Math.round(Number(document.getElementById(id)?.style.flexGrow))))
    const initial = await sizes()
    expect(initial.every((size) => size > 0)).toBe(true)
    const generate = page.getByRole("region", { name: "Result" }).getByRole("button", { name: /^Generate/ })
    await expect(generate).toHaveCount(0)

    await page.getByRole("button", { name: "Hide parameters" }).click()
    await expect.poll(sizes).toEqual([0, initial[0]! + initial[1]!, initial[2]])
    // the run controls move onto the stage while the parameters are hidden
    await expect(generate.first()).toBeVisible()
    await page.getByRole("button", { name: "Show parameters" }).click()
    await expect.poll(sizes).toEqual(initial)
    await expect(generate).toHaveCount(0)

    await page.getByRole("button", { name: "Hide history" }).click()
    await expect.poll(sizes).toEqual([initial[0], initial[1]! + initial[2]!, 0])
    await page.getByRole("button", { name: "Show history" }).click()
    await expect.poll(sizes).toEqual(initial)
  })
})

test.describe("tab strips on a phone", () => {
  test.use({ viewport: { width: 420, height: 860 } })

  for (const [route, label] of [
    ["extensions", "Extensions"],
    ["networks", "Network kinds"],
  ] as const) {
    test(`#/${route} shows its tabs as a select instead of cutting them`, async ({ page }) => {
      await page.goto(`/#/${route}`)
      const select = page.getByRole("main").getByRole("combobox", { name: label })
      await expect(select).toBeVisible()
      const overflowing = await page.evaluate(() =>
        [...document.querySelectorAll<HTMLElement>("[role=tablist]")]
          .filter((list) => list.getBoundingClientRect().width > 0 && getComputedStyle(list).visibility !== "hidden")
          .filter((list) => list.scrollWidth > list.clientWidth + 1)
          .map((list) => list.getAttribute("aria-label")),
      )
      expect(overflowing).toEqual([])
      // The select switches tabs like the strip.
      await select.click()
      const second = page.locator("[data-slot=select-item]").nth(1)
      const name = (await second.textContent())!.trim()
      await second.click()
      await expect(select).toHaveText(new RegExp(name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")))
    })
  }

  test("the tabs come back when there is room", async ({ page }) => {
    await page.goto("/#/extensions")
    await expect(page.getByRole("main").getByRole("combobox", { name: "Extensions" })).toBeVisible()
    await page.setViewportSize({ width: 1280, height: 860 })
    await expect(page.getByRole("tab", { name: "Backup and restore" })).toBeVisible()
    await expect(page.getByRole("main").getByRole("combobox", { name: "Extensions" })).toHaveCount(0)
  })
})

test.describe("hints and warnings", () => {
  test("the prompt placeholder names this platform's shortcuts", async ({ page }) => {
    await openGenerate(page)
    const prompt = await setPrompt(page, "x")
    await prompt.press("Backspace") // the placeholder shows in an empty prompt
    const generate = process.platform === "darwin" ? "⌘Enter" : "Ctrl+Enter"
    await expect(page.locator(".cm-placeholder").first()).toContainText(`${generate} to Generate`)
    await expect(page.getByRole("button", { name: /^Generate/ }).first()).toContainText(generate)
  })

  test("a LoRA that is not there ends the job with a warning", async ({ page }) => {
    await openGenerate(page)
    await setPrompt(page, "1girl, silver hair, <lora:doesnotexist:1>")
    await page.keyboard.press(`${mod}+Enter`)
    const toast = page.locator("[data-sonner-toast]").filter({ hasText: "Finished with warnings" })
    await expect(toast).toBeVisible({ timeout: 20_000 })
    await expect(toast).toContainText("LoRA not found or failed to load: doesnotexist")
  })
})

test.describe("schema rows on extension pages", () => {
  const actions = ["Build prompt", "Send to txt2img Prompt", "Replace txt2img Prompt", "Clear all fields"]

  test("an action row is one left-aligned group of same-sized buttons; status lines use the app's status style", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 })
    await page.goto("/#/ext/toolbox")
    const main = page.getByRole("main")
    const boxes = []
    for (const name of actions) boxes.push((await main.getByRole("button", { name, exact: true }).boundingBox())!)
    expect(new Set(boxes.map((box) => Math.round(box.height))).size).toBe(1)
    for (const [index, box] of boxes.slice(1).entries()) expect(box.x - (boxes[index]!.x + boxes[index]!.width)).toBeLessThan(16) // side by side, not spread out
    await main.getByRole("button", { name: "Build prompt" }).click()
    const status = main.locator("[data-tone=success]")
    await expect(status).toHaveText("Prompt built") // the "✅" became the icon and tone
  })

  test("two columns of fields stack on a phone", async ({ page }) => {
    for (const [width, sideBySide] of [
      [1280, true],
      [420, false],
    ] as const) {
      await page.setViewportSize({ width, height: 900 })
      await page.goto("/#/ext/toolbox")
      const series = page.getByRole("textbox", { name: "Series" })
      await expect(series).toBeVisible()
      // both boxes in one frame: measured one after the other they can straddle a late layout change
      const sameLine = async () =>
        page
          .getByRole("textbox", { name: "Quality" })
          .evaluate((quality, other) => Math.abs(quality.getBoundingClientRect().top - other!.getBoundingClientRect().top) < 2, await series.elementHandle())
      await expect.poll(sameLine, { message: `${width}px` }).toBe(sideBySide)
    }
  })
})

/** Toasts against the controls pinned to the bottom edge (`data-toast-obstacle`) they overlap, as messages. */
async function toastOverlaps(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const toasts = document.querySelectorAll<HTMLElement>("[data-sonner-toast]")
    // (a toast that timed out proves nothing)
    const problems: string[] = toasts.length ? [] : ["no toast"]
    for (const toast of toasts) {
      const t = toast.getBoundingClientRect()
      for (const obstacle of document.querySelectorAll<HTMLElement>("[data-toast-obstacle]")) {
        const o = obstacle.getBoundingClientRect()
        if (o.width === 0 || o.height === 0) continue
        // the close button sticks out of the toast's top-left corner
        const x = Math.min(t.right, o.right) - Math.max(t.left - 8, o.left)
        const y = Math.min(t.bottom, o.bottom) - Math.max(t.top - 8, o.top)
        if (x > 0 && y > 0) problems.push(`toast covers ${obstacle.getAttribute("aria-label") ?? obstacle.tagName}`)
      }
    }
    return problems
  })
}

test.describe("toasts keep clear of the bottom controls", () => {
  for (const viewport of [
    { width: 420, height: 860 },
    { width: 1280, height: 720 },
  ]) {
    test(`at ${viewport.width}×${viewport.height}`, async ({ page }) => {
      await page.setViewportSize(viewport)
      await openGenerate(page)
      await setPrompt(page, "1girl, <lora:doesnotexist:1>")
      await page.keyboard.press(`${mod}+Enter`)
      const toast = page.locator("[data-sonner-toast]").filter({ hasText: "Finished with warnings" })
      await expect(toast).toBeVisible({ timeout: 20_000 })
      await toast.hover() // keeps it open while it is measured
      if (viewport.width < 768) await expect(page.getByRole("navigation", { name: "Main navigation" })).toBeVisible()
      await expect.poll(() => toastOverlaps(page)).toEqual([])
    })
  }
})

test.describe("form rows", () => {
  test.use({ viewport: { width: 2560, height: 1440 } })

  test("a slider beside selects lines up its label and its track with theirs", async ({ page }) => {
    await openGenerate(page)
    const box = async (name: string) => (await page.locator("label", { hasText: new RegExp(`^${name}$`) }).first().boundingBox())!
    const [method, schedule, steps] = [await box("Sampling method"), await box("Schedule type"), await box("Sampling steps")]
    // one row: the three fields side by side, their labels on one line
    expect(Math.abs(method.y - schedule.y)).toBeLessThan(1)
    expect(Math.abs(method.y - steps.y)).toBeLessThan(1)
    expect(steps.x).toBeGreaterThan(schedule.x)
    const select = (await page.getByRole("combobox", { name: "Sampling method" }).boundingBox())!
    const track = (await page.locator("[data-slot=slider-track]").first().boundingBox())!
    expect(Math.abs(select.y + select.height / 2 - (track.y + track.height / 2))).toBeLessThan(2)
  })
})

test.describe("page container", () => {
  test.use({ viewport: { width: 1920, height: 1080 } })

  test("extension pages share the Extensions page's column", async ({ page }) => {
    const column = async (hash: string) => {
      await page.goto(`/#/${hash}`)
      const root = page.locator("main [data-scroll-root] > div").first()
      await expect(root).toBeVisible()
      const box = (await root.boundingBox())!
      const padding = await root.evaluate((element) => getComputedStyle(element).paddingLeft)
      return { left: Math.round(box.x), width: Math.round(box.width), padding }
    }
    expect(await column("ext/toolbox")).toEqual(await column("extensions"))
  })
})

test.describe("toasts keep clear of the canvas toolbars", () => {
  test.use({ viewport: { width: 1280, height: 720 } })

  test("img2img at 1280×720", async ({ page }) => {
    await openGenerate(page, "img2img")
    const png = await page.evaluate(() => {
      const canvas = document.createElement("canvas")
      canvas.width = 640
      canvas.height = 832
      const ctx = canvas.getContext("2d")!
      ctx.fillStyle = "#2a6df4"
      ctx.fillRect(0, 0, 640, 832)
      return canvas.toDataURL("image/png").split(",")[1]!
    })
    const chooser = page.waitForEvent("filechooser")
    await page.getByRole("button", { name: "Add a source image" }).click()
    await (await chooser).setFiles({ name: "source.png", mimeType: "image/png", buffer: Buffer.from(png, "base64") })
    await expect(page.getByRole("img", { name: /^Image editor, 640×832/ })).toBeVisible()
    const toolbar = page.getByRole("toolbar", { name: "Image", exact: true })
    await expect(toolbar).toBeVisible()
    // an action's toast (the mock's LCM compatibility note) while the source is on the stage
    await page.getByRole("combobox", { name: "Sampling method" }).click()
    await page.getByRole("option", { name: "LCM", exact: true }).click()
    const toast = page.locator("[data-sonner-toast]").filter({ hasText: "LCM uses the SGM Uniform schedule" })
    await expect(toast).toBeVisible({ timeout: 20_000 })
    await toast.hover()
    // the toolbar is under the toast column: without the lift the toast would cover it
    const bar = (await toolbar.boundingBox())!
    expect(bar.x + bar.width).toBeGreaterThan(1280 - 24 - 356)
    await expect.poll(() => toastOverlaps(page)).toEqual([])
  })
})

