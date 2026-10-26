/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "1" serves API and WebSocket from local fixtures (src/api/mock). */
  readonly VITE_MOCK?: string
  /** Mock mode: how many images the mock gallery holds (default 260; e.g. 10000 to measure scrolling). */
  readonly VITE_MOCK_GALLERY_SIZE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
