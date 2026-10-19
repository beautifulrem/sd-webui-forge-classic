/**
 * Extension management against the mock backend (src/api/mock/admin.ts), so the flows that need
 * the network or change the install on a real server run offline: install from a URL and from
 * the index, the update check with its progress, and restoring a backup behind its confirmation.
 * Each test also fails on console errors.
 */
import { expect, test } from "@playwright/test"
import { openRoute, watchConsole } from "./helpers"

test.describe("extension management", () => {
  test("installs from a URL, lists the new extension and refuses the same folder twice", async ({ page }) => {
    const console = watchConsole(page)
    await openRoute(page, "extensions?tab=url")
    const panel = page.getByRole("main")
    const url = panel.getByRole("textbox", { name: "Repository URL" })
    // a URL that is not a git remote is refused before anything is sent
    await url.fill("not a url")
    await url.blur()
    await expect(panel.getByText("Enter a git URL (https://, ssh://, git:// or git@host:path).")).toBeVisible()
    await expect(panel.getByRole("button", { name: "Install" })).toBeDisabled()

    await url.fill("https://github.com/example/sd-webui-e2e-extension.git")
    // the folder name comes from the URL
    await expect(panel.getByRole("textbox", { name: "Folder name" })).toHaveAttribute("placeholder", "sd-webui-e2e-extension")
    await panel.getByRole("button", { name: "Install" }).click()
    await expect(panel.getByRole("button", { name: "Installing…" })).toBeDisabled()
    await expect(panel.getByText("sd-webui-e2e-extension is installed.")).toBeVisible({ timeout: 10_000 })
    await expect(panel.getByRole("button", { name: "Reload UI" })).toBeVisible()

    // the same folder again: the server's conflict, shown in place
    await panel.getByRole("button", { name: "Install" }).click()
    await expect(panel.getByText("the folder sd-webui-e2e-extension already exists")).toBeVisible({ timeout: 10_000 })

    await page.getByRole("tab", { name: "Installed" }).click()
    await expect(page.getByRole("switch", { name: "sd-webui-e2e-extension" })).toBeChecked()
    console.assertClean()
  })

  test("loads the index and installs an extension from it", async ({ page }) => {
    const console = watchConsole(page)
    await openRoute(page, "extensions?tab=available")
    await page.getByRole("button", { name: "Load index" }).click()
    const list = page.getByRole("list", { name: "Available extensions" })
    const install = list.getByRole("button", { name: /^Install / }).first()
    await expect(install).toBeVisible({ timeout: 10_000 })
    const name = (await install.getAttribute("aria-label"))!.replace(/^Install /, "")
    await install.click()
    await expect(page.getByText(`${name} is installed`, { exact: true })).toBeVisible({ timeout: 10_000 })
    // the entry is marked installed: no Install button any more (installed ones are hidden by default)
    await expect(list.getByRole("button", { name: `Install ${name}`, exact: true })).toHaveCount(0)
    await page.getByRole("tab", { name: "Installed" }).click()
    await expect(page.getByRole("switch", { name, exact: true })).toBeVisible()
    console.assertClean()
  })

  test("checks for updates with progress, then offers the updates it found", async ({ page }) => {
    const console = watchConsole(page)
    await openRoute(page, "extensions")
    await page.getByRole("button", { name: "Check for updates" }).click()
    const progress = page.getByRole("progressbar", { name: "Checking for updates" })
    await expect(progress).toBeVisible()
    await expect(page.getByText(/^Checked \d+ of \d+/)).toBeVisible()
    await expect(page.getByRole("button", { name: "Stop" })).toBeVisible()
    // done: the bar goes away and a toast sums it up
    await expect(progress).toBeHidden({ timeout: 20_000 })
    await expect(page.getByText(/^\d+ updates available$|^Everything is up to date$/)).toBeVisible()
    // the mock reports an update for every extension with an even-length name
    await expect(page.getByRole("button", { name: "Update" }).first()).toBeVisible()
    console.assertClean()
  })

  test("restores a backup only after its confirmation, then reloads the UI", async ({ page }) => {
    const console = watchConsole(page)
    await openRoute(page, "extensions?tab=backup")
    await page.getByRole("listbox", { name: "Backups" }).getByRole("option").first().click()
    const restore = page.getByRole("main").getByRole("button", { name: "Restore extensions" })
    await expect(restore).toBeEnabled()

    // Cancel: nothing happens
    await restore.click()
    const confirm = page.getByRole("alertdialog", { name: "Restore the extensions of this backup?" })
    await expect(confirm).toBeVisible()
    await expect(confirm.getByText("Every extension is reset to its recorded commit, and those not in the backup are disabled. The UI reloads to apply it.")).toBeVisible()
    await confirm.getByRole("button", { name: "Cancel" }).click()
    await expect(confirm).toBeHidden()
    await expect(page.getByRole("alertdialog", { name: "Reloading the UI…" })).toHaveCount(0)

    // Confirmed: restored, then the UI reloads (the reconnect overlay, then a fresh page)
    await page.evaluate(() => Object.assign(window, { __beforeRestore: true }))
    await restore.click()
    await confirm.getByRole("button", { name: "Restore extensions" }).click()
    await expect(page.getByRole("alertdialog", { name: "Reloading the UI…" })).toBeVisible()
    await expect.poll(() => page.evaluate(() => "__beforeRestore" in window).catch(() => true), { timeout: 20_000 }).toBe(false)
    await expect(page.getByRole("listbox", { name: "Backups" })).toBeVisible()
    console.assertClean()
  })
})
