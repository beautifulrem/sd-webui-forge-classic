import { act, render, screen } from "@testing-library/react"
import { useTranslation } from "react-i18next"
import { afterEach, describe, expect, it, vi } from "vitest"
import { activateLocale, i18n } from "./index"

function Probe() {
  const { t } = useTranslation()
  return (
    <p>
      {t("Load more")} / {t("Sampling method")} / {t("Untranslated label")}
    </p>
  )
}

describe("locale switching", () => {
  afterEach(async () => {
    vi.restoreAllMocks()
    await activateLocale("en")
  })

  it("loads the chrome and server catalogs and re-renders without a reload", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ "Sampling method": "采样方法" }), { headers: { "Content-Type": "application/json" } }))
    render(<Probe />)
    expect(screen.getByText("Load more / Sampling method / Untranslated label")).toBeInTheDocument()

    await act(() => activateLocale("zh_CN"))
    expect(new URL((fetchSpy.mock.calls[0]![0] as Request).url).pathname).toBe("/api/v2/i18n/zh_CN")
    expect(i18n.language).toBe("zh-CN")
    expect(document.documentElement.lang).toBe("zh-CN")
    // chrome → server → the English key itself.
    expect(screen.getByText("加载更多 / 采样方法 / Untranslated label")).toBeInTheDocument()

    await act(() => activateLocale("en"))
    expect(screen.getByText("Load more / Sampling method / Untranslated label")).toBeInTheDocument()
  })
})
