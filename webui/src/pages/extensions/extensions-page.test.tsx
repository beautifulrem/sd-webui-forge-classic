import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { queryClient } from "@/api/queries"
import { i18n } from "@/i18n"
import ExtensionsPage from "./index"
import type { ConfigStateDetail, ExtensionIndex, ExtensionInfo, ExtensionList } from "./extensions-model"

const ext = (name: string, extra: Partial<ExtensionInfo> = {}): ExtensionInfo => ({
  name,
  enabled: true,
  builtin: false,
  remote: `https://github.com/u/${name}`,
  remote_url: `https://github.com/u/${name}`,
  branch: "main",
  version: "abcdef12",
  commit_hash: "abcdef1234",
  commit_url: null,
  commit_date: "2026-01-02T00:00:00Z",
  status: null,
  can_update: false,
  ...extra,
})

function listing(items: ExtensionInfo[], extra: Partial<ExtensionList> = {}): ExtensionList {
  return { items, disable_all: "none", access: true, restartable: true, disabled_by: null, index_url: "https://example.org/index.json", ...extra }
}

function renderPage(query?: string) {
  return render(
    <QueryClientProvider client={queryClient}>
      <ExtensionsPage route={{ name: "extensions", query: query ? new URLSearchParams(query) : undefined }} />
    </QueryClientProvider>,
  )
}

describe("extensions page", { timeout: 20_000 }, () => {
  let items: ExtensionInfo[]
  beforeEach(() => {
    items = [ext("tagger"), ext("browser", { enabled: false }), ext("sd_forge_lora", { builtin: true, remote: null, remote_url: null, version: null })]
    vi.spyOn(api.extensions, "list").mockImplementation(async () => listing(items))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("lists user and built-in extensions apart; built-in ones cannot be uninstalled", async () => {
    renderPage()
    expect(await screen.findByText("tagger")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Built-in/ })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /Built-in/ }))
    fireEvent.click(await screen.findByRole("button", { name: "Actions for sd_forge_lora" }))
    const menu = await screen.findByRole("menu")
    expect(within(menu).queryByRole("menuitem", { name: /Uninstall/ })).not.toBeInTheDocument()
  })

  it("stages switches and saves them as the disabled list", async () => {
    const configure = vi.spyOn(api.extensions, "configure").mockImplementation(async (body) => listing(items.map((e) => ({ ...e, enabled: !body.disabled?.includes(e.name) }))))
    renderPage()
    fireEvent.click(await screen.findByRole("switch", { name: "tagger" }))
    expect(await screen.findByText("1 switched, 0 to update")).toBeInTheDocument()
    expect(configure).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(configure).toHaveBeenCalledWith({ disabled: ["tagger", "browser"] }))
    await waitFor(() => expect(screen.queryByText("1 switched, 0 to update")).not.toBeInTheDocument())
  })

  it("checks for updates one extension at a time and offers the update", async () => {
    const check = vi.spyOn(api.extensions, "check").mockImplementation(async (name) => ({ ...items.find((e) => e.name === name)!, can_update: true, status: "new commits" }))
    renderPage()
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }))
    // Disabled and built-in extensions are skipped.
    await waitFor(() => expect(check).toHaveBeenCalledTimes(1))
    expect(check).toHaveBeenCalledWith("tagger")
    expect(await screen.findByText("Update available")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("checkbox", { name: "Update" }))
    expect(await screen.findByText("0 switched, 1 to update")).toBeInTheDocument()
  })

  it("explains when extension access is disabled", async () => {
    vi.spyOn(api.extensions, "list").mockResolvedValue(listing(items, { access: false }))
    renderPage()
    expect(await screen.findByText(/Extension management is disabled/)).toBeInTheDocument()
    expect(screen.getByRole("switch", { name: "tagger" })).toHaveAttribute("aria-disabled", "true")
  })

  it("validates the install form", async () => {
    const install = vi.spyOn(api.extensions, "install").mockResolvedValue(ext("sd-new"))
    renderPage("tab=url")
    const url = await screen.findByLabelText("Repository URL", {}, { timeout: 8000 })
    fireEvent.change(url, { target: { value: "not a url" } })
    fireEvent.blur(url)
    expect(await screen.findByText(/Enter a git URL/)).toBeInTheDocument()
    fireEvent.change(url, { target: { value: "https://github.com/u/sd-new.git" } })
    expect(screen.getByLabelText("Folder name")).toHaveAttribute("placeholder", "sd-new")
    fireEvent.click(screen.getByRole("button", { name: "Install" }))
    await waitFor(() => expect(install).toHaveBeenCalledWith({ url: "https://github.com/u/sd-new.git", branch: null, dirname: null }))
    expect(await screen.findByText("sd-new is installed.")).toBeInTheDocument()
  })

  it("opens the built-in group when a filter matches inside it, and labels switches of folders with spaces", async () => {
    items.push(ext("my ext", { remote: null, remote_url: null }))
    renderPage()
    const spaced = await screen.findByRole("switch", { name: "my ext" })
    const description = spaced.getAttribute("aria-describedby")
    expect(description && document.getElementById(description)).toBeTruthy()
    expect(screen.queryByText("sd_forge_lora")).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole("searchbox", { name: "Filter extensions" }), { target: { value: "lora" } })
    expect(await screen.findByRole("switch", { name: "sd_forge_lora" })).toBeInTheDocument()
  })

  it("shows install errors in the user's language", async () => {
    vi.spyOn(api.extensions, "install").mockRejectedValue(
      new ApiError(409, "the folder sd-new already exists", { detail: "the folder sd-new already exists", template: "the folder {{name}} already exists", params: { name: "sd-new" } }),
    )
    i18n.addResourceBundle("zh-CN", "chrome", { "the folder {{name}} already exists": "文件夹 {{name}} 已存在" }, true, true)
    await i18n.changeLanguage("zh-CN")
    try {
      renderPage("tab=url")
      const url = await screen.findByRole("textbox", { name: /git|URL|地址|仓库/i }, { timeout: 8000 })
      fireEvent.change(url, { target: { value: "https://github.com/u/sd-new.git" } })
      fireEvent.submit(url.closest("form")!)
      expect(await screen.findByText("文件夹 sd-new 已存在")).toBeInTheDocument()
    } finally {
      await i18n.changeLanguage("en")
    }
  })
})

const detail = (id: string): ConfigStateDetail => ({
  id,
  name: "Before update",
  created_at: "2026-09-30T12:00:00Z",
  webui_commit: "a1b2c3d4",
  extensions: 1,
  webui: { saved: { commit_hash: "a1b2c3d4" }, current: { commit_hash: "f6e7d8c9" } },
  items: [{ name: "tagger", builtin: false, saved: { commit_hash: "1111111111", enabled: true }, current: { commit_hash: "2222222222", enabled: true } }],
})

describe("backups", { timeout: 20_000 }, () => {
  beforeEach(() => {
    vi.spyOn(api.extensions, "list").mockResolvedValue(listing([ext("tagger")]))
    vi.spyOn(api.extensions.states, "list").mockResolvedValue([{ id: "a.json", name: "Before update", created_at: "2026-09-30T12:00:00Z", webui_commit: "a1b2c3d4", extensions: 1 }])
    vi.spyOn(api.extensions.states, "get").mockImplementation(async (id) => detail(id))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("restores only the extensions, after confirming", async () => {
    const restore = vi.spyOn(api.extensions.states, "restore").mockResolvedValue({ reload: true })
    renderPage("tab=backup")
    fireEvent.click(await screen.findByRole("button", { name: "Restore extensions" }, { timeout: 8000 }))
    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/Every extension is reset to its recorded commit/)).toBeInTheDocument()
    expect(screen.queryByRole("radio", { name: "WebUI" })).not.toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole("button", { name: "Restore extensions" }))
    await waitFor(() => expect(restore).toHaveBeenCalledWith("a.json"))
  })

  it("asks before deleting a backup", async () => {
    const remove = vi.spyOn(api.extensions.states, "remove").mockResolvedValue(undefined as never)
    renderPage("tab=backup")
    fireEvent.click(await screen.findByRole("button", { name: "Delete this backup" }, { timeout: 8000 }))
    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/“Before update” is deleted/)).toBeInTheDocument()
    expect(remove).not.toHaveBeenCalled()
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument())
    expect(remove).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Delete this backup" }))
    fireEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Delete" }))
    await waitFor(() => expect(remove).toHaveBeenCalledWith("a.json"))
  })
})

const indexOf = (count: number): ExtensionIndex => ({
  url: "https://example.org/index.json",
  fetched_at: null,
  tags: { script: "scripts", installed: "already installed" },
  items: Array.from({ length: count }, (_, i) => ({
    name: `ext-${i}`,
    url: `https://github.com/u/ext-${i}`,
    description: `Extension number ${i}`,
    tags: ["script"],
    added: i % 2 === 0 ? `2024-01-${String((i % 28) + 1).padStart(2, "0")}` : null,
    created_at: null,
    commit_time: null,
    stars: i,
    order: i,
    installed: false,
  })),
})

describe("available extensions", { timeout: 20_000 }, () => {
  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(800)
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(1000)
    vi.spyOn(api.extensions, "list").mockResolvedValue(listing([]))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("loads the index into one virtualized list in the page scroll, and installs", async () => {
    const index = vi.spyOn(api.extensions, "index").mockResolvedValue(indexOf(500))
    const install = vi.spyOn(api.extensions, "install").mockResolvedValue(ext("ext-0"))
    const { container } = renderPage("tab=available")
    fireEvent.click(await screen.findByRole("button", { name: "Load index" }, { timeout: 8000 }))
    await waitFor(() => expect(index).toHaveBeenCalled())
    const list = await screen.findByRole("list", { name: "Available extensions" })
    // Virtualized: far fewer rows than entries; no scroller of its own.
    const rows = within(list).getAllByRole("listitem")
    expect(rows.length).toBeGreaterThan(0)
    expect(rows.length).toBeLessThan(100)
    expect(container.querySelectorAll(".overflow-y-auto")).toHaveLength(1)
    // Newest first: dated entries before undated ones.
    expect(within(rows[0]!).getByText("ext-26")).toBeInTheDocument()
    fireEvent.change(screen.getByRole("searchbox", { name: "Search extensions" }), { target: { value: "ext-7" } })
    fireEvent.click(await screen.findByRole("button", { name: "Install ext-7" }))
    await waitFor(() => expect(install).toHaveBeenCalledWith({ url: "https://github.com/u/ext-7" }))
  })
})
