import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { type Favorite, type FavoritesPage, promptToolsApi } from "../api"
import { TEST_CONFIG } from "../testing"
import FavoritesPanel from "./FavoritesPanel"
import GeneratePanel from "./GeneratePanel"
import GroupsPanel from "./GroupsPanel"
import { groupTagText } from "./helpers"

const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>

const panelProps = (value: string, setValue = vi.fn()) => ({ value, setValue, kind: "positive" as const, label: "Prompt", disabled: false, config: TEST_CONFIG, close: vi.fn() })

afterEach(() => {
  vi.restoreAllMocks()
  queryClient.clear()
})

describe("quick tag groups", () => {
  it("write tags like an accepted completion", () => {
    expect(groupTagText("hanten_(clothes)", TEST_CONFIG.tags)).toBe("hanten \\(clothes\\)")
    expect(groupTagText("^_^", { ...TEST_CONFIG.tags, underscore_exclusions: ["^_^"] })).toBe("^_^")
    expect(groupTagText("long_hair", { ...TEST_CONFIG.tags, replace_underscores: false })).toBe("long_hair")
  })

  it("add an escaped tag and recognise it to remove it again", async () => {
    vi.spyOn(promptToolsApi, "groups").mockResolvedValue([{ name: "服装", groups: [{ name: "上衣", color: null, tags: [{ tag: "hanten_(clothes)", label: "半缠" }] }] }])
    const setValue = vi.fn()
    const { rerender } = render(<GroupsPanel {...panelProps("1girl", setValue)} />, { wrapper })
    const button = await screen.findByRole("button", { name: /半缠/ })
    expect(button).toHaveAttribute("aria-pressed", "false")
    fireEvent.click(button)
    expect(setValue).toHaveBeenLastCalledWith("1girl, hanten \\(clothes\\)")
    rerender(<GroupsPanel {...panelProps("1girl, hanten \\(clothes\\)", setValue)} />)
    expect(screen.getByRole("button", { name: /半缠/ })).toHaveAttribute("aria-pressed", "true")
    fireEvent.click(screen.getByRole("button", { name: /半缠/ }))
    expect(setValue).toHaveBeenLastCalledWith("1girl")
  }, 20_000)
})

const favorite = (id: string, text: string, fields: Partial<Favorite> = {}): Favorite => ({ id, collection_id: null, kind: "prompt", text, name: "", created: 0, position: 0, ...fields })

describe("favorites", () => {
  function setup() {
    vi.spyOn(promptToolsApi.collections, "list").mockResolvedValue([
      { id: null, name: "", created: 0, count: 1 },
      { id: "c1", name: "Faces", created: 0, count: 2 },
    ])
    const list = vi.spyOn(promptToolsApi.favorites, "list").mockImplementation(async (query): Promise<FavoritesPage> => {
      if (query.cursor) return { items: [favorite("f3", "third prompt")], next_cursor: null, total: 3 }
      return { items: [favorite("f1", "first prompt", { name: "First" }), favorite("f2", "smile, blush", { kind: "tags" })], next_cursor: "page2", total: 3 }
    })
    const update = vi.spyOn(promptToolsApi.favorites, "update").mockImplementation(async (id) => favorite(id, "x"))
    const setValue = vi.fn()
    const props = panelProps("1girl", setValue)
    render(<FavoritesPanel {...props} />, { wrapper })
    return { list, update, setValue, props }
  }

  it("asks the server for this prompt's kind and tag snippets, page by page", async () => {
    const { list } = setup()
    await screen.findByText("first prompt")
    expect(list).toHaveBeenCalledWith(expect.objectContaining({ kind: ["prompt", "tags"], collection: undefined, cursor: null }), expect.anything())
    fireEvent.click(screen.getByRole("button", { name: "Load more" }))
    await screen.findByText("third prompt")
    expect(list).toHaveBeenLastCalledWith(expect.objectContaining({ cursor: "page2" }), expect.anything())
    expect(screen.queryByRole("button", { name: "Load more" })).toBeNull()
  }, 20_000)

  it("uses a prompt, appends tags", async () => {
    const { setValue, props } = setup()
    fireEvent.click(await screen.findByRole("button", { name: /first prompt/ }))
    expect(setValue).toHaveBeenLastCalledWith("first prompt")
    expect(props.close).toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: /smile, blush/ }))
    expect(setValue).toHaveBeenLastCalledWith("1girl, smile, blush")
  }, 20_000)

  it("moves a favorite to a collection from its menu", async () => {
    const { update } = setup()
    const row = (await screen.findByText("first prompt")).closest("li")!
    act(() => within(row).getByRole("button", { name: "Move to a collection" }).click())
    const target = await screen.findByRole("menuitemradio", { name: "Faces" })
    act(() => target.click())
    await waitFor(() => expect(update).toHaveBeenCalledWith("f1", { collection_id: "c1" }))
  }, 20_000)

  it("filters by collection", async () => {
    const { list } = setup()
    await screen.findByText("first prompt")
    act(() => screen.getByRole("combobox", { name: "Collection" }).click())
    const faces = await screen.findByRole("option", { name: "Faces (2)" })
    act(() => faces.click())
    await waitFor(() => expect(list).toHaveBeenLastCalledWith(expect.objectContaining({ collection: "c1" }), expect.anything()))
  }, 20_000)

  it("imports a file and links the export", async () => {
    setup()
    const imported = vi.spyOn(promptToolsApi.favorites, "import").mockResolvedValue({ added: 2 })
    await screen.findByText("first prompt")
    expect(screen.getByRole("link", { name: "Export favorites" })).toHaveAttribute("href", expect.stringContaining("/prompt-tools/favorites-export"))
    const input = document.querySelector<HTMLInputElement>('input[type="file"]')!
    const data = { version: 1, favorites: [{ text: "a" }, { text: "b" }] }
    const file = new File([JSON.stringify(data)], "favorites.json", { type: "application/json" })
    fireEvent.change(input, { target: { files: [file] } })
    await waitFor(() => expect(imported).toHaveBeenCalledWith(data))
  }, 20_000)
})

describe("writing a prompt", () => {
  it("sends the description, then replaces or extends the prompt with the edited answer", async () => {
    const generate = vi.spyOn(promptToolsApi, "generate").mockResolvedValue("1girl, reading, cherry blossoms")
    const setValue = vi.fn()
    const props = panelProps("masterpiece", setValue)
    const { unmount } = render(<GeneratePanel {...props} />, { wrapper })
    fireEvent.change(screen.getByLabelText("Describe the image"), { target: { value: "  a girl reading under a cherry tree " } })
    fireEvent.click(screen.getByRole("checkbox"))
    fireEvent.click(screen.getByRole("button", { name: "Write" }))
    const written = await screen.findByLabelText("Written prompt")
    expect(generate).toHaveBeenCalledWith({ description: "a girl reading under a cherry tree", prompt: "masterpiece", negative: false })
    fireEvent.change(written, { target: { value: "1girl, reading" } })
    fireEvent.click(screen.getByRole("button", { name: "Add to the prompt" }))
    expect(setValue).toHaveBeenLastCalledWith("masterpiece, 1girl, reading")
    expect(props.close).toHaveBeenCalled()
    unmount()

    render(<GeneratePanel {...panelProps("", setValue)} kind="negative" />, { wrapper })
    expect(screen.queryByRole("checkbox")).toBeNull() // nothing to build on
    fireEvent.change(screen.getByLabelText("Describe the image"), { target: { value: "blur" } })
    fireEvent.click(screen.getByRole("button", { name: "Write" }))
    fireEvent.click(await screen.findByRole("button", { name: "Replace the prompt" }))
    expect(generate).toHaveBeenLastCalledWith({ description: "blur", prompt: "", negative: true })
    expect(setValue).toHaveBeenLastCalledWith("1girl, reading, cherry blossoms")
  })

  it("shows the service's error", async () => {
    vi.spyOn(promptToolsApi, "generate").mockRejectedValue(new Error("translation service unreachable"))
    render(<GeneratePanel {...panelProps("")} />, { wrapper })
    fireEvent.change(screen.getByLabelText("Describe the image"), { target: { value: "a cat" } })
    fireEvent.click(screen.getByRole("button", { name: "Write" }))
    expect(await screen.findByRole("alert")).toHaveTextContent("translation service unreachable")
  })
})
