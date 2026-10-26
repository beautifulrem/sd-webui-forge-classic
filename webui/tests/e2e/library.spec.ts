/** Gallery, PNG Info and the Queue against the mock backend. */
import { expect, type Page, test } from "@playwright/test"
import { injectFaults } from "./helpers"

async function openGallery(page: Page) {
  await page.goto("/#/gallery")
  const grid = page.getByRole("list", { name: "Images" })
  await expect(grid.getByRole("listitem").first()).toBeVisible()
  return grid
}

test.describe("gallery", () => {
  test("a failing server shows the error, and the toolbar stops saying Loading…", async ({ page }) => {
    await injectFaults(page, [{ path: "^/gallery(/facets)?$", status: 500 }])
    await page.goto("/#/gallery")
    await expect(page.getByText("Gallery unavailable")).toBeVisible({ timeout: 20_000 })
    await expect(page.getByRole("main").getByText("Loading…")).toHaveCount(0)
    await expect(page.getByRole("main").locator("[aria-busy=true]")).toHaveCount(0)
  })

  test("filters by search and opens the viewer with the parameters beside the image", async ({ page }) => {
    const grid = await openGallery(page)
    await page.getByRole("searchbox", { name: "Search the gallery" }).fill("lighthouse")
    await expect(page.getByText(/images match/)).toBeVisible()
    await expect(grid.getByRole("button", { name: /^lighthouse at dusk/ }).first()).toBeVisible()
    await expect(grid.getByRole("button", { name: /^a red fox/ })).toHaveCount(0)

    await grid.getByRole("button", { name: /^lighthouse at dusk/ }).first().click()
    const details = page.getByRole("complementary", { name: "Image details" })
    await expect(details).toBeVisible()
    await expect(details.getByRole("button", { name: "PNG Info" })).toBeVisible()
    await expect(details.getByText("lighthouse at dusk, volumetric light").first()).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(details).toBeHidden()
  })

  test("selects two images and compares them", async ({ page }) => {
    const grid = await openGallery(page)
    const tiles = grid.getByRole("listitem")
    const bulk = page.getByRole("toolbar", { name: "Selected images" })
    for (const index of [0, 1]) {
      await tiles.nth(index).hover()
      await tiles.nth(index).getByRole("checkbox").click()
      await expect(bulk).toContainText(`${index + 1} selected`)
    }
    await bulk.getByRole("button", { name: "Compare" }).click()
    const dialog = page.getByRole("dialog")
    const slider = dialog.getByRole("slider", { name: "Comparison divider" })
    await expect(slider).toBeVisible()
    await expect(slider).toHaveAttribute("aria-valuenow", "50")
    // The dialog moves the focus when it opens: retry until the key reaches the handle.
    await expect(async () => {
      await slider.press("Home")
      await expect(slider).toHaveAttribute("aria-valuenow", "0", { timeout: 500 })
    }).toPass()
  })

  test("sends an image's parameters to txt2img through PNG Info", async ({ page }) => {
    const grid = await openGallery(page)
    await grid.getByRole("button", { name: /^a red fox/ }).first().click()
    await page.getByRole("complementary", { name: "Image details" }).getByRole("button", { name: "PNG Info" }).click()
    await expect(page).toHaveURL(/#\/png-info\?file=/)
    await expect(page.getByRole("tab", { name: /All metadata/ })).toBeVisible()
    await page.getByRole("button", { name: "To txt2img" }).click()
    await expect(page).toHaveURL(/#\/txt2img$/)
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toContainText("a red fox in fresh snow")
  })
})

/** Client-side navigation: the mock backend lives in the page, a reload would reset it. */
async function go(page: Page, hash: string) {
  await page.evaluate((target) => (location.hash = target), hash)
}

test.describe("queue", () => {
  test("lines jobs up while paused, reorders them from the keyboard and resumes", async ({ page }) => {
    await page.goto("/#/queue")
    await page.getByRole("button", { name: "Pause" }).click()
    await expect(page.getByRole("button", { name: "Resume" })).toBeVisible()

    await go(page, "#/txt2img")
    const prompt = page.getByRole("textbox", { name: "Prompt", exact: true })
    for (const text of ["first job", "second job"]) {
      await prompt.click()
      await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A")
      await page.keyboard.type(text)
      await page.getByRole("button", { name: "Add to queue" }).click()
      await expect(page.getByText(/Added to the queue as #/).last()).toBeVisible()
    }

    await go(page, "#/queue")
    const queued = page.getByRole("list", { name: "Queued jobs" }).getByRole("listitem")
    await expect(queued).toHaveCount(2)
    await expect(queued.nth(0)).toContainText("first job")

    const grip = queued.nth(1).getByRole("button", { name: /^Move job/ })
    await grip.focus()
    await page.keyboard.press("Space")
    await page.keyboard.press("ArrowUp")
    await page.keyboard.press("Space")
    await expect(queued.nth(0)).toContainText("second job")

    await page.getByRole("button", { name: "Resume" }).click()
    await expect(page.getByRole("button", { name: "Pause" })).toBeVisible()
    await expect(page.getByRole("list", { name: "Job history" }).getByText("second job")).toBeVisible({ timeout: 20_000 })
  })
})
