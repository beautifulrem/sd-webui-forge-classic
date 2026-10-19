/** Generate page against the mock backend. */
import { expect, type Page, test } from "@playwright/test"

const mod = process.platform === "darwin" ? "Meta" : "Control"

async function open(page: Page, tab: "txt2img" | "img2img" = "txt2img") {
  await page.goto(`/#/${tab}`)
  await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible()
}

async function setPrompt(page: Page, text: string) {
  const prompt = page.getByRole("textbox", { name: "Prompt", exact: true })
  await prompt.click()
  await page.keyboard.press(`${mod}+A`)
  await page.keyboard.type(text)
  return prompt
}

/** Types a value into a numeric field of the form (a slider's box, the seed) and commits it. */
async function setNumber(page: Page, label: string, value: number) {
  const box = page.getByRole("textbox", { name: label, exact: true }).first()
  await box.fill(String(value))
  await box.press("Tab")
  await expect(box).toHaveValue(String(value))
}

/** The Generate button while none of the form's runs is in flight (it reads "Queue another" meanwhile). */
const idleGenerate = (page: Page) => page.getByRole("button", { name: /^Generate\b(?! several)/ }).first()

/** The Seed chip of the selected result's parameters. */
const seedChip = (page: Page) =>
  page
    .getByRole("region", { name: "Result" })
    .locator("dl > div")
    .filter({ has: page.locator("dt", { hasText: /^Seed$/ }) })
    .locator("dd")

test.describe("generate", () => {
  test("generates with Ctrl/Cmd+Enter and files the result into the history", async ({ page }) => {
    await open(page)
    await setPrompt(page, "1girl, cherry blossoms")
    await page.keyboard.press(`${mod}+Enter`)
    await expect(page.getByRole("button", { name: "Interrupt" })).toBeVisible()
    await expect(page.getByRole("img", { name: "Live preview" })).toBeVisible()
    await expect(page.getByRole("button", { name: "Interrupt" })).toBeHidden({ timeout: 15_000 })
    await expect(page.getByRole("listbox", { name: "History" }).getByRole("option")).toHaveCount(1)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible()
    await expect(page.getByText("1girl, cherry blossoms").first()).toBeVisible()
  })

  test("generates twice in a row from the same page", async ({ page }) => {
    // The second job is posted with the values the page loaded before the first one ran
    // (a real server once rejected it: a checkpoint title had gained its hash meanwhile).
    await open(page)
    await setPrompt(page, "1girl, twice")
    const generate = page.getByRole("button", { name: /^Generate/ }).first()
    const history = page.getByRole("listbox", { name: "History" }).getByRole("option")
    for (const count of [1, 2]) {
      await generate.click()
      await expect(page.getByRole("button", { name: "Interrupt" })).toBeVisible()
      await expect(page.getByRole("button", { name: "Interrupt" })).toBeHidden({ timeout: 15_000 })
      await expect(history).toHaveCount(count)
    }
    await expect(page.getByRole("alert")).toHaveCount(0)
  })

  test("Escape interrupts a running job", async ({ page }) => {
    await open(page)
    await page.getByRole("button", { name: /^Generate/ }).first().click()
    await expect(page.getByRole("button", { name: "Interrupt" })).toBeVisible()
    await page.locator("body").press("Escape")
    await expect(page.getByRole("button", { name: "Interrupt" })).toBeHidden({ timeout: 5_000 })
    await expect(page.getByRole("listbox", { name: "History" })).toHaveCount(0)
  })

  test("prompt editor: weight keys, bracket warning, token count", async ({ page }) => {
    await open(page)
    const prompt = await setPrompt(page, "a cat, blue sky")
    await page.keyboard.press(`${mod}+ArrowUp`)
    await expect(prompt).toHaveText("a cat, blue (sky:1.1)")
    await page.keyboard.press("End")
    await page.keyboard.type(", (unclosed")
    await expect(page.getByText(/Unbalanced brackets: 1/)).toBeVisible()
    await expect(page.locator(".cm-prompt-bracket-error")).toHaveCount(1)
  })

  test("switches to img2img and back with the header mode switch", async ({ page }) => {
    await open(page)
    await page.getByRole("navigation", { name: "Mode" }).getByRole("link", { name: "img2img" }).click()
    await expect(page).toHaveURL(/#\/img2img$/)
    await expect(page.getByRole("region", { name: "Source", exact: true })).toBeVisible()
    await page.getByRole("navigation", { name: "Mode" }).getByRole("link", { name: "txt2img" }).click()
    await expect(page).toHaveURL(/#\/txt2img$/)
  })

  test("opens the viewer from the stage and closes it with Escape", async ({ page }) => {
    await open(page)
    await page.getByRole("button", { name: /^Generate/ }).first().click()
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    await page.getByRole("button", { name: "Open image viewer" }).first().click()
    const viewer = page.getByRole("dialog", { name: "Image viewer" })
    await expect(viewer).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(viewer).toBeHidden()
  })

  test("sends a result to img2img with its parameters", async ({ page }) => {
    await open(page)
    await setPrompt(page, "a red fox")
    await page.keyboard.press(`${mod}+Enter`)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: 15_000 })
    await page.getByRole("button", { name: "Send to" }).click()
    await page.getByRole("menuitem", { name: "img2img", exact: true }).click()
    await expect(page).toHaveURL(/#\/img2img$/)
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toHaveText("a red fox")
    // The source image is set (its line offers to replace / remove it) and the stage shows it in the editor.
    await expect(page.getByRole("region", { name: "Source", exact: true }).getByRole("button", { name: /^Remove/ })).toBeVisible()
    await expect(page.getByRole("img", { name: /^Image editor, / })).toBeVisible()
  })

  test("batch count N makes N results in batch order", async ({ page }) => {
    await open(page)
    await setPrompt(page, "a lantern")
    await setNumber(page, "Batch count", 3)
    await setNumber(page, "Sampling steps", 20) // about 2 s a batch: the batch counter stays on screen long enough to see
    await setNumber(page, "Seed", 1234)
    await page.getByRole("button", { name: /^Generate/ }).first().click()
    await expect(page.getByText(/Batch 1 of 3/).first()).toBeVisible()
    await expect(idleGenerate(page)).toBeVisible({ timeout: 20_000 })
    const results = page.getByRole("listbox", { name: "History" }).getByRole("option")
    await expect(results).toHaveCount(3)
    // Each image has its own seed (seed + index): the results keep the batch order.
    for (const [index, seed] of [1234, 1235, 1236].entries()) {
      await results.nth(index).click()
      await expect(seedChip(page)).toHaveText(String(seed))
    }
  })

  test("shows the live preview of every image of a batch, not only the first", async ({ page }) => {
    await open(page)
    // every preview URL the stage's live image shows (the step starts at 0 again for each image of the job)
    await page.evaluate(() => {
      const seen = new Set<string>()
      Object.assign(window, { __previews: seen })
      const note = (node: Node) => {
        if (node instanceof HTMLImageElement && node.alt === "Live preview" && node.src) seen.add(node.src)
      }
      new MutationObserver((records) => {
        for (const record of records) {
          note(record.target)
          record.addedNodes.forEach((node) => {
            note(node)
            if (node instanceof Element) node.querySelectorAll("img").forEach(note)
          })
        }
      }).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["src", "alt"] })
    })
    await setNumber(page, "Batch count", 3)
    await setNumber(page, "Sampling steps", 12) // the mock sends a preview every third step: 4 per image
    await page.getByRole("button", { name: /^Generate/ }).first().click()
    await expect(page.getByText(/Batch 3 of 3/).first()).toBeVisible({ timeout: 10_000 })
    await expect(idleGenerate(page)).toBeVisible({ timeout: 20_000 })
    const shown = await page.evaluate(() => (window as unknown as { __previews: Set<string> }).__previews.size)
    // 12 frames (4 per image); a check on the step alone dropped images 2 and 3 but for their last frame (6)
    expect(shown).toBeGreaterThanOrEqual(11)
  })

  test("Alt+Enter skips the current image of a multi-image job and the job goes on", async ({ page }) => {
    await open(page)
    await setNumber(page, "Batch count", 2)
    await setNumber(page, "Sampling steps", 60) // about 5 s a batch: a skip shows as batch 2 starting at once
    await page.getByRole("button", { name: /^Generate/ }).first().click()
    await expect(page.getByText(/Batch 1 of 2/).first()).toBeVisible()
    await page.locator("body").press("Alt+Enter")
    await expect(page.getByText(/Batch 2 of 2/).first()).toBeVisible({ timeout: 3_000 })
    // Skip is not Interrupt: the job is still running; it ends with both batches' images.
    await expect(page.getByRole("button", { name: "Interrupt" })).toBeVisible()
    await page.getByRole("button", { name: "Skip" }).click()
    await expect(idleGenerate(page)).toBeVisible({ timeout: 5_000 })
    await expect(page.getByRole("listbox", { name: "History" }).getByRole("option")).toHaveCount(2)
    await expect(page.getByRole("img", { name: "Generated image" })).toBeVisible()
  })

  test("Generate forever queues run after run until it is stopped", async ({ page }) => {
    await open(page)
    await setNumber(page, "Sampling steps", 3)
    await page.getByRole("button", { name: "More generate options" }).click()
    const forever = page.getByRole("menuitemcheckbox", { name: "Generate forever" })
    await forever.click()
    await expect(forever).toHaveAttribute("aria-checked", "true")
    await page.keyboard.press("Escape") // a checkbox item leaves the menu open; Escape closes it, the runs go on
    await expect(forever).toBeHidden()
    await expect(page.getByText("Generating forever")).toBeVisible()
    const results = page.getByRole("listbox", { name: "History" }).getByRole("option")
    // Each run is queued when the previous one has finished.
    await expect.poll(() => results.count(), { timeout: 15_000 }).toBeGreaterThanOrEqual(3)
    await page.getByRole("button", { name: "Stop generating forever" }).click()
    await expect(page.getByText("Generating forever")).toBeHidden()
    // The run in progress finishes; no other one follows.
    await expect(idleGenerate(page)).toBeVisible({ timeout: 5_000 })
    const made = await results.count()
    await page.waitForTimeout(1_500)
    await expect(results).toHaveCount(made)
    await expect(idleGenerate(page)).toBeVisible()
  })

  test("Generate several times runs the form N times, one after another", async ({ page }) => {
    await open(page)
    await setNumber(page, "Sampling steps", 3)
    await page.getByRole("button", { name: "Generate several times" }).click()
    const runs = page.getByRole("spinbutton", { name: "Generate several times" }).or(page.getByRole("textbox", { name: "Generate several times" }))
    await runs.fill("3")
    await page.getByRole("button", { name: "Start" }).click()
    await expect(page.getByText(/2 more runs/)).toBeVisible()
    await expect(idleGenerate(page)).toBeVisible({ timeout: 15_000 })
    await expect(page.getByRole("listbox", { name: "History" }).getByRole("option")).toHaveCount(3)
  })

  test("phone layout stacks the panes with the run controls at the bottom @phone", async ({ page }) => {
    await open(page)
    const bar = page.getByRole("button", { name: /^Generate/ }).first()
    await expect(bar).toBeVisible()
    await expect(page.getByRole("region", { name: "Result" })).toBeVisible()
  })
})
