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

  test("phone layout stacks the panes with the run controls at the bottom @phone", async ({ page }) => {
    await open(page)
    const bar = page.getByRole("button", { name: /^Generate/ }).first()
    await expect(bar).toBeVisible()
    await expect(page.getByRole("region", { name: "Result" })).toBeVisible()
  })
})
