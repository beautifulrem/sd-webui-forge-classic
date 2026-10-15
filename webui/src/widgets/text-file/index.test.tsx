import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { CustomNode } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import TextFile from "./index"
import { fileLines } from "./lines"
import zh from "./locales/zh-CN.json"

const PROPS = { accept: ".txt,.csv", lines: 2, max_lines: 20 }
const NODE: CustomNode = { type: "custom", name: "prompt_txt", label: "List of prompt inputs", widget: "text-file", props: PROPS, default: "" }

function setup(value = "") {
  const onChange = vi.fn()
  render(<TextFile value={value} onChange={onChange} props={PROPS} node={NODE} disabled={false} id="prompts" label="List of prompt inputs" />)
  return onChange
}

describe("text-file widget", () => {
  it("edits the text", () => {
    const onChange = setup("a cat")
    fireEvent.change(document.getElementById("prompts")!, { target: { value: "a cat\na dog" } })
    expect(onChange).toHaveBeenLastCalledWith("a cat\na dog")
  })

  it("loads a file in the browser, each line stripped", async () => {
    const onChange = setup()
    const picker = screen.getByTestId("text-file-picker") as HTMLInputElement
    expect(picker.accept).toBe(".txt,.csv")
    const file = new File(["  a cat  \r\n--prompt 'a dog' --steps 10\n"], "prompts.txt", { type: "text/plain" })
    fireEvent.change(picker, { target: { files: [file] } })
    await waitFor(() => expect(onChange).toHaveBeenCalledWith("a cat\n--prompt 'a dog' --steps 10\n"))
  })

  it("strips like Python's decode(errors='ignore') + strip()", () => {
    expect(fileLines("\tx\uFFFD \r\n\n y")).toBe("x\n\ny")
  })

  it("has a zh-CN translation for every chrome string", () => {
    const sources = import.meta.glob<string>(["./*.tsx", "!./*.test.*"], { query: "?raw", import: "default", eager: true })
    const keys = Object.values(sources).flatMap(widgetKeys)
    expect(keys.length).toBeGreaterThan(0)
    expect(keys.filter((key) => !(zh as Record<string, string>)[key])).toEqual([])
  })
})
