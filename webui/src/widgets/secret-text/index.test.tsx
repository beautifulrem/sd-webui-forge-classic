import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { CustomNode } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import SecretText from "./index"
import zh from "./locales/zh-CN.json"
import source from "./index.tsx?raw"

const NODE: CustomNode = { type: "custom", name: "api_key", label: "api_key", widget: "secret-text", props: {}, default: "" }

describe("secret-text widget", () => {
  it("is masked until shown, and reports edits", () => {
    const onChange = vi.fn()
    render(<SecretText value="" onChange={onChange} props={{ placeholder: "Saved keys are not shown" }} node={NODE} disabled={false} id="key" label="api_key" />)
    const input = screen.getByLabelText("api_key")
    expect(input).toHaveAttribute("type", "password")
    expect(input).toHaveAttribute("placeholder", "Saved keys are not shown")
    expect(input).toHaveAttribute("autocomplete", "off")
    fireEvent.change(input, { target: { value: "abc" } })
    expect(onChange).toHaveBeenCalledWith("abc")
    fireEvent.click(screen.getByRole("button", { name: "Show" }))
    expect(input).toHaveAttribute("type", "text")
    expect(screen.getByRole("button", { name: "Hide" })).toHaveAttribute("aria-pressed", "true")
  })

  it("translates every chrome string", () => {
    const catalog = zh as Record<string, string>
    expect(widgetKeys(source).filter((key) => !catalog[key])).toEqual([])
  })
})
