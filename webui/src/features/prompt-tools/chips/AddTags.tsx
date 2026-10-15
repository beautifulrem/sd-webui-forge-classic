/**
 * The input at the end of the tag view: type tags (separated by commas);
 * the last one is completed from the tag list like in the text editor
 * (category colours, translations, post counts). ↓/↑ pick a suggestion and
 * Enter takes it; Enter without one adds what was typed.
 */
import { useQuery } from "@tanstack/react-query"
import { useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Combobox, ComboboxChipsInput, ComboboxContent, ComboboxItem, ComboboxList } from "@/components/ui/combobox"
import type { PromptKind } from "@/features/prompt-editor/facets"
import { useDebouncedValue } from "@/features/prompt-editor/use-debounced-value"
import { type CompletionItem, type PromptToolsConfig, promptToolsApi } from "../api"
import { categoryColor, compactCount } from "../completion/colors"
import { tagText } from "../completion/insert"
import { appendChips, type PromptDoc, serializePrompt } from "./parse"

type AddTagsProps = { doc: PromptDoc; setValue: (value: string) => void; disabled: boolean; empty: boolean; config: PromptToolsConfig; kind: PromptKind }

const SUGGESTIONS = 8

/** The typed tags (comma separated, trimmed) as prompt text. */
function typedTags(text: string): string {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .join(", ")
}

/** The text with its last (partly typed) tag replaced by `tag`. */
function withLastTag(text: string, tag: string): string {
  const comma = text.lastIndexOf(",")
  return comma < 0 ? tag : `${text.slice(0, comma + 1)} ${tag}`
}

export function AddTags({ doc, setValue, disabled, empty, config, kind }: AddTagsProps) {
  const { t } = useTranslation()
  const [text, setText] = useState("")
  const [open, setOpen] = useState(false)
  const highlighted = useRef<CompletionItem | undefined>(undefined)
  const negative = kind === "negative"
  const completing = config.tags.enabled && (!negative || config.tags.negative)
  const word = useDebouncedValue(text.split(",").at(-1)!.trim(), 150)
  const suggestions = useQuery({
    queryKey: ["prompt-tools", "suggest", word, negative],
    queryFn: ({ signal }) => promptToolsApi.complete({ mode: "tag", q: word, limit: SUGGESTIONS, negative }, signal),
    enabled: completing && word.length > 0,
    staleTime: 60_000,
  })
  const items = word && text.trim() ? (suggestions.data ?? []) : []

  const add = (typed: string) => {
    const added = typedTags(typed)
    if (!added) return
    setValue(appendChips(doc, added))
    setText("")
  }

  const choose = (item: CompletionItem) => {
    add(withLastTag(text, tagText(item, config.tags)))
    // counted like an accepted completion (frequency ranking)
    const name = item.type === "lora" || item.type === "chant" ? (item.label ?? item.text) : item.text
    void promptToolsApi.recordUse({ name, type: item.type, negative }).catch(() => {})
  }

  return (
    <Combobox<CompletionItem>
      items={items}
      filter={null}
      value={null}
      open={open && items.length > 0}
      onOpenChange={setOpen}
      inputValue={text}
      onInputValueChange={(next, details) => {
        // choosing an item writes its label into the input: the tag is added instead
        if (details.reason === "input-change" || details.reason === "input-clear") {
          setText(next)
          setOpen(true)
        }
      }}
      onValueChange={(item) => {
        if (item) choose(item)
      }}
      onItemHighlighted={(item) => {
        highlighted.current = item
      }}
      itemToStringLabel={(item) => item.text}
      disabled={disabled}
    >
      <ComboboxChipsInput
        disabled={disabled}
        spellCheck={false}
        aria-label={t("Add tags")}
        placeholder={empty ? t("Add tags, separated by commas") : t("Add tags")}
        onKeyDown={(event) => {
          if (event.nativeEvent.isComposing) return
          if (event.key === "Enter" && !highlighted.current) {
            event.preventDefault()
            add(text)
          } else if (event.key === "Backspace" && !text && doc.chips.length > 0) {
            // Backspace in the empty input takes the last tag back for editing
            event.preventDefault()
            const last = doc.chips.at(-1)!
            setText(last.raw)
            setValue(serializePrompt({ ...doc, chips: doc.chips.slice(0, -1), seps: doc.seps.slice(0, -1), tail: "" }))
          }
        }}
        className="h-6 w-full min-w-0 rounded bg-transparent px-1 text-xs placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
      />
      <ComboboxContent className="w-72 min-w-0">
        <ComboboxList>
          {(item: CompletionItem) => {
            const color = item.type === "tag" || item.type === "extra" ? categoryColor(config.tags.source, item.category) : null
            return (
              <ComboboxItem key={`${item.type}:${item.text}`} value={item} className="pr-2 text-xs">
                <span className="min-w-0 flex-1 truncate font-medium" style={color ? { color } : undefined}>
                  {tagText({ ...item, category: null }, { ...config.tags, escape_parentheses: false })}
                </span>
                {item.translation ? <span className="max-w-28 shrink truncate text-muted-foreground">{item.translation}</span> : null}
                {item.count ? <span className="tabular w-9 shrink-0 text-right text-muted-foreground">{compactCount(item.count)}</span> : null}
              </ComboboxItem>
            )
          }}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  )
}
