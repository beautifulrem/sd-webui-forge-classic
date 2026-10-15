import { fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"
import { SliderRow } from "./SliderRow"

type FakeSliderProps = { value: number; onValueChange?: (value: number) => void; onValueCommitted?: (value: number) => void }

// The slider's pointer handling is base-ui's; here it shows its value and fakes drag steps and the drag's end
vi.mock("@/components/ui/slider", () => ({
  Slider: ({ value, onValueChange, onValueCommitted }: FakeSliderProps) => (
    <div data-testid="slider" data-value={value}>
      <button onClick={() => onValueChange?.(0.42)}>drag to 0.42</button>
      <button onClick={() => onValueChange?.(0.5)}>drag to 0.5</button>
      <button onClick={() => onValueCommitted?.(0.5)}>release at 0.5</button>
    </div>
  ),
}))

function Harness({ onCommit, initial = 1 }: { onCommit: (value: number) => void; initial?: number }) {
  const [value, setValue] = useState(initial)
  return (
    <>
      <SliderRow
        label="Weight"
        value={value}
        minimum={0}
        maximum={2}
        step={0.05}
        onCommit={(next) => {
          onCommit(next)
          setValue(next)
        }}
      />
      <button onClick={() => setValue(1.5)}>set from outside</button>
    </>
  )
}

describe("SliderRow", () => {
  it("shows a drag locally and commits once, when it ends", () => {
    const onCommit = vi.fn()
    render(<Harness onCommit={onCommit} />)
    fireEvent.click(screen.getByRole("button", { name: "drag to 0.42" }))
    fireEvent.click(screen.getByRole("button", { name: "drag to 0.5" }))
    expect(screen.getByTestId("slider")).toHaveAttribute("data-value", "0.5")
    expect(onCommit).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "release at 0.5" }))
    expect(onCommit).toHaveBeenCalledTimes(1)
    expect(onCommit).toHaveBeenCalledWith(0.5)
  })

  it("follows the value set elsewhere after a drag that was not committed", () => {
    render(<Harness onCommit={vi.fn()} />)
    fireEvent.click(screen.getByRole("button", { name: "drag to 0.42" }))
    fireEvent.click(screen.getByRole("button", { name: "set from outside" }))
    expect(screen.getByTestId("slider")).toHaveAttribute("data-value", "1.5")
  })

  it("commits a typed value", () => {
    const onCommit = vi.fn()
    render(<Harness onCommit={onCommit} />)
    const box = screen.getByRole("textbox", { name: "Weight" })
    fireEvent.change(box, { target: { value: "1.25" } })
    fireEvent.blur(box)
    expect(onCommit).toHaveBeenCalledWith(1.25)
  })
})
