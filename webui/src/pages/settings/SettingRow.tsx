/**
 * One settings item: the control (schema renderer: label, widget, modified
 * dot = reset to the built-in default), its help (notes, links, choice
 * descriptions, default order, a third-party option's HTML), effect badges, the unsaved marker, and the row actions (refresh the
 * choices, pin to quick settings, revert the unsaved edit).
 */
import { useQuery } from "@tanstack/react-query"
import { ExternalLink, Lock, Pin, PinOff, RefreshCw, Undo2 } from "lucide-react"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { useStore } from "zustand"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { MODEL_BAR_KEYS, QUICKSETTINGS_KEY, setPinned, useIsPinned } from "@/features/quicksettings/pinned"
import { cn } from "@/lib/utils"
import { runAction } from "@/ui/actions"
import { jsonEqual } from "@/ui/conditions"
import { RenderNode } from "@/ui/Renderer"
import { SafeHtml } from "@/ui/RichText"
import { safeUrl } from "@/ui/rich-text"
import type { SettingNode } from "@/ui/schema"
import { serverStatusQuery } from "./queries"
import { setSettingsMode } from "./settings-filters"
import type { SettingItem } from "./settings-model"
import { discardChanges, SETTINGS_FORM, SETTINGS_OWNER, settingsForm, useSettingsEditor } from "./settings-store"


function RowAction(props: { label: string; onClick: () => void; children: React.ReactNode; pressed?: boolean; disabled?: boolean }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={props.label}
            aria-pressed={props.pressed}
            disabled={props.disabled}
            onClick={props.onClick}
            className={cn("text-muted-foreground", props.pressed && "text-primary")}
          />
        }
      >
        {props.children}
      </TooltipTrigger>
      <TooltipContent>{props.label}</TooltipContent>
    </Tooltip>
  )
}

function useUnsaved(key: string): boolean {
  const value = useStore(settingsForm, (s) => s.values[SETTINGS_OWNER]?.[key])
  const saved = useSettingsEditor((s) => s.saved[key])
  const known = useSettingsEditor((s) => Object.hasOwn(s.saved, key))
  return known && !jsonEqual(value, saved)
}

/** Why a setting cannot be changed (not repeated per row when the whole page is frozen: a banner says so). */
function LockedNote({ reason }: { reason: string }) {
  const { t } = useTranslation()
  const frozen = useQuery({ ...serverStatusQuery, select: (status) => status.settings_frozen }).data
  if (frozen) return null
  return (
    <p className="flex items-center gap-1 text-foreground/80">
      <Lock className="size-3 shrink-0" aria-hidden />
      <span>
        {t("Locked")}: {t(reason)}
      </span>
    </p>
  )
}

/** The former quick settings hint popup: pins are set on each row, the Pinned filter lists them. */
function QuickSettingsHint() {
  const { t } = useTranslation()
  return (
    <p>
      {t("Pin a setting with the pin button next to it to add it here.")}{" "}
      <button type="button" className="text-primary underline-offset-2 hover:underline rounded-sm focus-visible:underline focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:outline-none" onClick={() => setSettingsMode("pinned")}>
        {t("Show pinned settings")}
      </button>
    </p>
  )
}

/** `[label, description]` of a choice setting's described choices, in choice order. */
function choiceDescriptions(node: SettingNode): [string, string][] {
  if (node.type !== "choice" && node.type !== "multichoice") return []
  const hints = node.choice_hints ?? {}
  return node.choices.flatMap((choice) => (hints[choice.label] ? [[choice.label, hints[choice.label]!] as [string, string]] : []))
}

/** Effect badges and help under a setting's control. */
function SettingHelp({ node }: { node: SettingNode }) {
  const { t } = useTranslation()
  const badges: string[] = []
  if (node.needs_restart) badges.push("Requires restart")
  else if (node.needs_reload_ui) badges.push("Requires Reload UI")
  const quick = node.name === QUICKSETTINGS_KEY
  const choiceHints = choiceDescriptions(node)
  const defaultOrder = node.default_order ?? []
  const hasHelp =
    (node.notes?.length ?? 0) > 0 || (node.links?.length ?? 0) > 0 || choiceHints.length > 0 || defaultOrder.length > 0 || Boolean(node.html) || Boolean(node.locked) || quick
  if (!hasHelp && badges.length === 0) return null
  return (
    <div className="flex flex-col gap-1 text-xs leading-snug text-muted-foreground">
      {node.locked ? <LockedNote reason={node.locked} /> : null}
      {node.notes?.map((note, index) => <p key={index}>{t(note)}</p>)}
      {quick ? <QuickSettingsHint /> : null}
      {choiceHints.length > 0 ? (
        <ul className="flex flex-col gap-0.5">
          {choiceHints.map(([label, hint]) => (
            <li key={label}>
              <span className="font-medium text-foreground/80">{t(label)}</span>: {t(hint)}
            </li>
          ))}
        </ul>
      ) : null}
      {defaultOrder.length > 0 ? (
        <div>
          <p>{t("Default order")}:</p>
          <ol className="list-decimal pl-5">
            {defaultOrder.map((name) => (
              <li key={name}>{name}</li>
            ))}
          </ol>
        </div>
      ) : null}
      {node.html ? <SafeHtml html={t(node.html)} className="[&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2 hover:[&_a]:decoration-2" /> : null}
      {(node.links?.length ?? 0) > 0 || badges.length > 0 ? (
        <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
          {badges.map((badge) => (
            <Badge key={badge} variant="secondary" className="h-4.5 px-1.5 text-[0.68rem] font-normal">
              {t(badge)}
            </Badge>
          ))}
          {node.links?.map((link) => (
            <a
              key={link.url}
              href={safeUrl(link.url) ?? undefined}
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex items-center gap-0.5 text-primary underline-offset-2 hover:underline rounded-sm focus-visible:underline focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:outline-none"
            >
              {t(link.label)}
              <ExternalLink className="size-3" aria-hidden />
            </a>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function SettingActions({ node }: { node: SettingNode }) {
  const { t } = useTranslation()
  const key = node.name
  const pinned = useIsPinned(key)
  const unsaved = useUnsaved(key)
  const [refreshing, setRefreshing] = useState(false)
  const label = t(node.label || key)
  return (
    <div className="flex shrink-0 items-center gap-0.5 pt-0.5 opacity-70 transition-opacity group-focus-within/setting:opacity-100 group-hover/setting:opacity-100">
      {unsaved ? (
        <RowAction label={t("Revert {{label}}", { label })} onClick={() => discardChanges([key])}>
          <Undo2 />
        </RowAction>
      ) : null}
      {node.refresh ? (
        <RowAction
          label={t("Refresh the choices of {{label}}", { label })}
          disabled={refreshing}
          onClick={() => {
            setRefreshing(true)
            void runAction({ store: settingsForm, tab: SETTINGS_FORM }, SETTINGS_OWNER, `refresh:${key}`).finally(() => setRefreshing(false))
          }}
        >
          <RefreshCw className={cn(refreshing && "animate-spin")} />
        </RowAction>
      ) : null}
      {MODEL_BAR_KEYS.has(key) ? null : (
        <RowAction
          label={pinned ? t("Unpin {{label}} from quick settings", { label }) : t("Pin {{label}} to quick settings", { label })}
          pressed={pinned}
          onClick={() => void setPinned(key, !pinned)}
        >
          {pinned ? <PinOff /> : <Pin />}
        </RowAction>
      )}
    </div>
  )
}

export function SettingView({ node }: { node: SettingNode }) {
  const unsaved = useUnsaved(node.name)
  const { t } = useTranslation()
  return (
    <div
      data-setting={node.name}
      className={cn(
        "group/setting relative -mx-3 flex flex-col gap-2 rounded-xl px-3 py-3 transition-colors",
        unsaved ? "bg-primary/[0.06]" : "hover:bg-muted/40",
      )}
    >
      {unsaved ? (
        <span aria-hidden className="absolute inset-y-3 left-0 w-0.5 rounded-full bg-primary" />
      ) : null}
      <div className="flex items-start gap-2">
        <div className="min-w-0 max-w-2xl flex-1">
          <RenderNode node={node} />
        </div>
        <SettingActions node={node} />
      </div>
      <SettingHelp node={node} />
      <div className="flex items-center gap-2">
        <code className="font-mono text-[0.68rem] text-muted-foreground select-all">{node.name}</code>
        {unsaved ? (
          <span className="text-[0.68rem] font-medium text-primary">{t("Unsaved")}</span>
        ) : null}
      </div>
    </div>
  )
}

export function SettingItemView({ item }: { item: SettingItem }) {
  const { t } = useTranslation()
  switch (item.kind) {
    case "setting":
      return <SettingView node={item.node} />
    case "row":
      return (
        <div className="grid gap-x-6 sm:grid-cols-2">
          {item.nodes.map((node) => (
            <SettingView key={node.name} node={node} />
          ))}
        </div>
      )
    case "html":
      // The catalogs key a help block by its exact markup (translations keep the markup)
      return <SafeHtml html={t(item.node.text)} className="py-2 text-sm text-muted-foreground [&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2 hover:[&_a]:decoration-2" />
    case "separator":
      return <Separator className="my-2" />
  }
}
