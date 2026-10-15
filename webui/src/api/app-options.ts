/**
 * Typed access to the settings the client reads from `Bootstrap.options`
 * (`GET /app`). The server sends a subset of the options; a missing key reads
 * as the A1111 default given here. Kept fresh by `options.changed`, which
 * invalidates the bootstrap query.
 */
import { useQuery } from "@tanstack/react-query"
import { bootstrapQuery, keys, queryClient } from "./queries"
import type { Bootstrap } from "./types"

/** Options the generate view reads, with their defaults (modules/shared_options.py). */
export const CLIENT_OPTION_DEFAULTS = {
  sd_model_checkpoint: null as string | null,
  forge_preset: "sd",
  forge_additional_modules: [] as string[],
  forge_unet_storage_dtype: "Automatic",
  ctrl_enter_interrupt: false,
  show_progress_in_title: true,
  show_progressbar: true,
  live_previews_enable: true,
  prevent_screen_sleep_during_generation: true,
  notification_audio: true,
  notification_volume: 100,
  return_grid: true,
  js_modal_lightbox: true,
  js_modal_lightbox_initially_zoomed: true,
  js_live_preview_in_modal_lightbox: false,
  js_modal_lightbox_gamepad: false,
  js_modal_lightbox_gamepad_repeat: 250,
  send_seed: true,
  send_size: true,
  disable_token_counters: false,
  include_styles_into_token_counters: true,
}

export type ClientOptions = typeof CLIENT_OPTION_DEFAULTS
export type ClientOptionKey = keyof ClientOptions

function pick<K extends ClientOptionKey>(options: Record<string, unknown> | undefined, key: K): ClientOptions[K] {
  const fallback = CLIENT_OPTION_DEFAULTS[key]
  const value = options?.[key]
  if (value === undefined || value === null) return fallback
  // A value of the wrong type (an older server) reads as the default.
  if (fallback !== null && typeof value !== typeof fallback) return fallback
  if (Array.isArray(fallback) && !Array.isArray(value)) return fallback
  return value as ClientOptions[K]
}

/** One option, reactive. */
export function useClientOption<K extends ClientOptionKey>(key: K): ClientOptions[K] {
  const { data } = useQuery(bootstrapQuery)
  return pick(data?.options, key)
}

/** One option from the query cache (for event handlers and non-React code). */
export function clientOption<K extends ClientOptionKey>(key: K): ClientOptions[K] {
  return pick(queryClient.getQueryData<Bootstrap>(keys.app)?.options, key)
}
