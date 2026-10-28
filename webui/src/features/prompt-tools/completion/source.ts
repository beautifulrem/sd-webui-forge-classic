/**
 * The tag autocomplete completion source (lazy chunk, loaded on the first
 * keystroke in a prompt editor): finds what is being typed, asks the server
 * (`/prompt-tools/complete`, cached per query), and inserts the accepted
 * item following the `tac_*` settings. Server order is kept (`filter:
 * false`): it already ranks by match, popularity and the user's usage.
 */
import { type Completion, type CompletionContext, type CompletionResult, startCompletion } from "@codemirror/autocomplete"
import type { EditorView } from "@codemirror/view"
import { api } from "@/api/endpoints"
import { completionAcceptKey, type DecoratedCompletion } from "@/features/prompt-editor/completion-registry"
import { promptKind, type PromptScope, promptScope } from "@/features/prompt-editor/facets"
import { t } from "@/i18n"
import { cn } from "@/lib/utils"
import { type CompletionItem, type CompletionQuery, type PromptToolsConfig, promptToolsApi } from "../api"
import { loadConfig } from "../queries"
import { categoryClass, categoryName, compactCount } from "./colors"
import { type CompletionTarget, completionTarget, tokenEnd } from "./context"
import { commonPrefix, insertion } from "./insert"

/** Wait this long before asking the server for an uncached query (typing fast), unless `tac_delayTime` says otherwise. */
export const FETCH_DELAY_MS = 40
const CACHE_SIZE = 400
const CACHE_TTL_MS = 60_000

const cache = new Map<string, { at: number; items: CompletionItem[] }>()

/** Forget cached completions (settings, networks or wildcards changed). */
export function clearCompletionCache() {
  cache.clear()
}

function cacheKey(query: CompletionQuery): string {
  return JSON.stringify([query.mode, query.q, query.negative, query.wildcard ?? "", query.limit ?? 0])
}

function cached(key: string): CompletionItem[] | null {
  const hit = cache.get(key)
  if (!hit || Date.now() - hit.at > CACHE_TTL_MS) return null
  // most recently used last
  cache.delete(key)
  cache.set(key, hit)
  return hit.items
}

function remember(key: string, items: CompletionItem[]) {
  cache.set(key, { at: Date.now(), items })
  while (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value!)
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

async function fetchItems(query: CompletionQuery, context: CompletionContext, delay: number): Promise<CompletionItem[] | null> {
  const key = cacheKey(query)
  const hit = cached(key)
  if (hit) return hit
  if (delay > 0) await sleep(delay)
  if (context.aborted) return null
  const controller = new AbortController()
  context.addEventListener("abort", () => controller.abort(), { onDocChange: true })
  try {
    const items = await promptToolsApi.complete(query, controller.signal)
    remember(key, items)
    return items
  } catch {
    return null
  }
}

/** `tac_activeIn_*`: completion in this editor's form and kind of field. */
export function activeIn(scope: PromptScope, config: PromptToolsConfig): boolean {
  const tags = config.tags
  if (scope.form === "txt2img" && !tags.txt2img) return false
  if (scope.form === "img2img" && !tags.img2img) return false
  return !scope.extension || tags.third_party
}

function enabled(target: CompletionTarget, config: PromptToolsConfig): boolean {
  const tags = config.tags
  switch (target.mode) {
    case "wildcard":
    case "wildcard-value":
      return tags.wildcards
    case "lora":
      return tags.loras
    case "embedding":
      return tags.embeddings
    case "chant":
      return tags.chants
    case "network":
      return tags.loras || tags.embeddings || tags.chants
    default:
      return true
  }
}

function element(tag: string, className: string, text?: string): HTMLElement {
  const node = document.createElement(tag)
  node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

const TYPE_LABELS: Partial<Record<CompletionItem["type"], () => string>> = {
  lora: () => t("LoRA"),
  embedding: () => t("Embedding"),
  chant: () => t("Chant"),
  wildcard: () => t("Wildcard"),
}

/** Translation, kind and post count after the label. */
function decoration(item: CompletionItem): Node {
  const box = element("span", "ml-auto flex min-w-0 shrink-0 items-center gap-2 pl-3 text-xs text-muted-foreground")
  if (item.translation) box.append(element("span", "max-w-40 truncate", item.translation))
  if (item.uses) box.append(element("span", "size-1.5 shrink-0 rounded-full bg-primary/70"))
  const kind = TYPE_LABELS[item.type]?.()
  if (kind) box.append(element("span", "rounded bg-muted px-1 py-px text-[0.65rem] leading-tight font-medium", kind))
  if (item.count) box.append(element("span", "tabular w-10 text-right", compactCount(item.count)))
  return box
}

function wikiUrl(source: string, tag: string): string {
  return source === "e621" ? `https://e621.net/wiki_pages/show_or_new?title=${encodeURIComponent(tag)}` : `https://danbooru.donmai.us/wiki_pages/${encodeURIComponent(tag)}`
}

/** The info panel beside the selected option. */
function info(item: CompletionItem, config: PromptToolsConfig): (() => Node) | undefined {
  const tags = config.tags
  const rows: [string, string][] = []
  const category = categoryName(tags.source, item.category)
  if ((item.type === "tag" || item.type === "extra") && category) rows.push([t("Category"), category])
  if (item.count) rows.push([t("Posts"), item.count.toLocaleString()])
  if (item.alias) rows.push([t("Alias"), item.alias])
  if (item.translation) rows.push([t("Translation"), item.translation])
  if (item.meta && item.type !== "wildcard") rows.push([item.type === "lora" ? t("Base model") : item.type === "wildcard-value" ? t("Wildcard") : t("Note"), item.meta])
  if (item.weight !== null && item.weight !== undefined) rows.push([t("Weight"), String(item.weight)])
  if (item.keywords) rows.push([t("Activation text"), item.keywords])
  if (item.uses) rows.push([t("Uses"), String(item.uses)])
  const chant = item.type === "chant" ? item.text : null
  if (!rows.length && !item.preview && !item.description && !chant) return undefined
  return () => {
    const box = element("div", "flex max-w-72 flex-col gap-2 p-1 text-xs")
    if (item.preview) {
      const image = element("img", "max-h-48 w-full rounded-md bg-checker object-contain") as HTMLImageElement
      image.src = api.thumbUrl(item.preview, 256)
      image.alt = ""
      image.loading = "lazy"
      box.append(image)
    }
    if (item.description) box.append(element("p", "line-clamp-4 text-muted-foreground", item.description))
    if (chant) box.append(element("p", "line-clamp-6 font-mono text-[0.7rem] break-words", chant))
    if (rows.length) {
      const list = element("dl", "grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5")
      for (const [term, value] of rows) list.append(element("dt", "text-muted-foreground", term), element("dd", "min-w-0 break-words", value))
      box.append(list)
    }
    if (tags.wiki_links && (item.type === "tag" || item.type === "extra")) {
      const link = element("a", "text-primary underline-offset-2 hover:underline", t("Open the wiki page")) as HTMLAnchorElement
      link.href = wikiUrl(tags.source, item.text)
      link.target = "_blank"
      link.rel = "noreferrer noopener"
      box.append(link)
    }
    return box
  }
}

function tagLabel(text: string, config: PromptToolsConfig): string {
  return config.tags.replace_underscores && !config.tags.underscore_exclusions.includes(text) ? text.replaceAll("_", " ") : text
}

function label(item: CompletionItem, config: PromptToolsConfig): string {
  switch (item.type) {
    case "tag":
    case "extra":
      // `tac_alias_onlyShowAlias`: the alias that matched stands for the tag (the tag is still inserted)
      return tagLabel(config.tags.only_alias && item.alias ? item.alias : item.text, config)
    case "lora":
    case "embedding":
    case "chant":
      return item.label ?? item.text
    case "wildcard":
      return `__${item.text}__`
    default:
      return item.text
  }
}

function detail(item: CompletionItem, config: PromptToolsConfig): string | undefined {
  if (item.alias) return config.tags.only_alias ? undefined : `← ${item.alias}`
  if (item.type === "lora" && item.label && item.label !== item.text) return item.text
  return undefined
}

/** Inserts `item` at the current target (recomputed: the document may have moved on). */
function apply(view: EditorView, item: CompletionItem, config: PromptToolsConfig, common: string | undefined) {
  const state = view.state
  const doc = state.doc.toString()
  const cursor = state.selection.main.head
  const target = completionTarget(doc, cursor)
  if (!target) return
  const result = insertion({ doc, target, to: tokenEnd(doc, target), item, settings: config.tags, key: completionAcceptKey(), common })
  view.dispatch({
    changes: result.changes,
    selection: { anchor: result.cursor },
    userEvent: "input.complete",
    scrollIntoView: true,
  })
  // a partial wildcard path is not a use yet
  const partial = item.type === "wildcard" && !result.changes.at(-1)!.insert.endsWith("__")
  if (!partial) {
    const name = item.type === "lora" || item.type === "chant" ? (item.label ?? item.text) : item.text
    void promptToolsApi.recordUse({ name, type: item.type, negative: state.facet(promptKind) === "negative" }).catch(() => {})
  }
  if (result.reopen) queueMicrotask(() => startCompletion(view))
}

function toCompletion(item: CompletionItem, config: PromptToolsConfig, common: string | undefined): DecoratedCompletion {
  const completion: DecoratedCompletion = {
    label: label(item, config),
    detail: detail(item, config),
    info: info(item, config),
    // CodeMirror's option rows are blocks: a flex row puts the decoration on the right
    className: cn("flex items-center", item.type === "tag" || item.type === "extra" ? categoryClass(config.tags.source, item.category) : ""),
    decorate: () => decoration(item),
    apply: (view: EditorView) => apply(view, item, config, common),
  }
  return completion
}

/** Where the typed text appears in a label (highlighted; the server matched it). */
export function matchRanges(label: string, query: string): number[] {
  const needle = query.replace(/^\*/, "").replace(/\\/g, "").trim().toLowerCase().replace(/_/g, " ")
  if (!needle) return []
  const at = label.toLowerCase().replace(/_/g, " ").indexOf(needle)
  return at < 0 ? [] : [at, at + needle.length]
}

/** The completion source registered with every prompt editor. */
export async function promptToolsCompletions(context: CompletionContext): Promise<CompletionResult | null> {
  const config = await loadConfig()
  if (!config?.tags.enabled || context.aborted) return null
  const negative = context.state.facet(promptKind) === "negative"
  if (negative && !config.tags.negative) return null
  if (!activeIn(context.state.facet(promptScope), config)) return null
  const target = completionTarget(context.state.doc.toString(), context.pos)
  if (!target || !enabled(target, config)) return null
  // a tag needs a character typed (or Ctrl+Space)
  if (target.mode === "tag" && !target.query.trim() && !context.explicit) return null
  const items = await fetchItems(
    { mode: target.mode, q: target.query, negative, limit: target.mode === "tag" ? config.tags.max_results : undefined, wildcard: target.wildcard },
    context,
    config.tags.delay ?? FETCH_DELAY_MS,
  )
  if (!items?.length || context.aborted) return null
  const common = target.mode === "wildcard" ? commonPrefix(items.map((item) => item.text)) : undefined
  const options: Completion[] = items.map((item) => toCompletion(item, config, common))
  return { from: target.from, to: context.pos, options, filter: false, getMatch: (completion) => matchRanges(completion.label, target.query) }
}
