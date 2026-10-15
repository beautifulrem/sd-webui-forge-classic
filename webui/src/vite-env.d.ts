/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "1" serves API and WebSocket from local fixtures (src/api/mock). */
  readonly VITE_MOCK?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
