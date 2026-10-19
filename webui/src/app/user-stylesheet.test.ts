import { afterEach, describe, expect, it } from "vitest"
import { applyUserStylesheet, USER_CSS_ID } from "./user-stylesheet"

const link = () => document.getElementById(USER_CSS_ID) as HTMLLinkElement | null

describe("user.css", () => {
  afterEach(() => link()?.remove())

  it("links user.css after every other stylesheet when the server lists it", () => {
    const appStyles = document.createElement("style")
    document.head.append(appStyles)
    applyUserStylesheet(["notification.mp3", "user.css"])
    const added = link()
    expect(added?.rel).toBe("stylesheet")
    expect(added?.getAttribute("href")).toBe("/api/v2/user-assets/user.css")
    // in <body>: later in document order than anything in <head>, including CSS added after it
    expect(added?.parentElement).toBe(document.body)
    const lateChunk = document.createElement("link")
    document.head.append(lateChunk)
    expect(lateChunk.compareDocumentPosition(added!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    appStyles.remove()
    lateChunk.remove()
  })

  it("adds it once and removes it when the file is gone", () => {
    applyUserStylesheet(["user.css"])
    applyUserStylesheet(["user.css"])
    expect(document.querySelectorAll(`#${USER_CSS_ID}`)).toHaveLength(1)
    applyUserStylesheet(["notification.mp3"])
    expect(link()).toBeNull()
  })

  it("links nothing without user.css", () => {
    applyUserStylesheet([])
    applyUserStylesheet(undefined)
    expect(link()).toBeNull()
  })
})
