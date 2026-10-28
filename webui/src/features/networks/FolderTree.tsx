/**
 * Folder tree with item counts (a WAI-ARIA tree: one tab stop, ↑/↓ move,
 * →/← expand/collapse or go to the child/parent, Enter/Space select,
 * Home/End). "All" is the root; selecting a folder filters the grid to it and
 * its subfolders.
 */
import { ChevronRight, Folder, FolderOpen, Library } from "@/components/icons"
import { useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import type { NetworkFolder } from "@/api/types"
import { cn } from "@/lib/utils"

type Row = { path: string; name: string; count: number; depth: number; hasChildren: boolean; parent: string | null }

function flatten(nodes: readonly NetworkFolder[], expanded: ReadonlySet<string>, depth = 1, parent: string | null = "", out: Row[] = []): Row[] {
  for (const node of nodes) {
    const children = node.children ?? []
    out.push({ path: node.path, name: node.name, count: node.count, depth, hasChildren: children.length > 0, parent })
    if (children.length && expanded.has(node.path)) flatten(children, expanded, depth + 1, node.path, out)
  }
  return out
}

const ancestors = (path: string) => path.split("/").slice(0, -1).map((_, i, parts) => parts.slice(0, i + 1).join("/"))

export function FolderTree({ folders, total, selected, onSelect, className }: { folders: readonly NetworkFolder[]; total: number; selected: string; onSelect: (path: string) => void; className?: string }) {
  const { t } = useTranslation()
  const id = useId()
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set(ancestors(selected)))
  const [focused, setFocused] = useState(selected)
  const list = useRef<HTMLUListElement>(null)

  // A folder chosen elsewhere (a chip, a new search) is revealed.
  const [shown, setShown] = useState(selected)
  if (shown !== selected) {
    setShown(selected)
    setFocused(selected)
    const missing = ancestors(selected).filter((path) => !expanded.has(path))
    if (missing.length) setExpanded(new Set([...expanded, ...missing]))
  }

  const rows: Row[] = [{ path: "", name: t("All"), count: total, depth: 0, hasChildren: false, parent: null }, ...flatten(folders, expanded)]
  const focusedIndex = Math.max(0, rows.findIndex((row) => row.path === focused))
  const rowId = (path: string) => `${id}-${encodeURIComponent(path || "/")}`

  const toggle = (path: string, open?: boolean) =>
    setExpanded((old) => {
      const next = new Set(old)
      if (open ?? !next.has(path)) next.add(path)
      else next.delete(path)
      return next
    })

  const focusRow = (index: number) => {
    const row = rows[Math.max(0, Math.min(rows.length - 1, index))]
    if (!row) return
    setFocused(row.path)
    list.current?.querySelector<HTMLElement>(`#${CSS.escape(rowId(row.path))}`)?.focus()
  }

  const onKeyDown = (event: React.KeyboardEvent) => {
    const row = rows[focusedIndex]
    if (!row) return
    const keys: Record<string, () => void> = {
      ArrowDown: () => focusRow(focusedIndex + 1),
      ArrowUp: () => focusRow(focusedIndex - 1),
      Home: () => focusRow(0),
      End: () => focusRow(rows.length - 1),
      ArrowRight: () => {
        if (!row.hasChildren) return
        if (expanded.has(row.path)) focusRow(focusedIndex + 1)
        else toggle(row.path, true)
      },
      ArrowLeft: () => {
        if (row.hasChildren && expanded.has(row.path)) toggle(row.path, false)
        else if (row.parent !== null) focusRow(rows.findIndex((other) => other.path === row.parent))
      },
      Enter: () => onSelect(row.path),
      " ": () => onSelect(row.path),
    }
    const run = keys[event.key]
    if (run) {
      event.preventDefault()
      run()
    }
  }

  return (
    <ul ref={list} role="tree" aria-label={t("Folders")} onKeyDown={onKeyDown} className={cn("flex flex-col gap-px py-1 text-[0.8rem]", className)}>
      {rows.map((row) => {
        const isSelected = row.path === selected
        const open = expanded.has(row.path)
        const Icon = row.depth === 0 ? Library : open ? FolderOpen : Folder
        return (
          <li
            key={row.path || "/"}
            id={rowId(row.path)}
            role="treeitem"
            aria-level={row.depth + 1}
            aria-selected={isSelected}
            aria-expanded={row.hasChildren ? open : undefined}
            tabIndex={row.path === rows[focusedIndex]?.path ? 0 : -1}
            onFocus={() => setFocused(row.path)}
            onClick={() => onSelect(row.path)}
            className={cn(
              "group/tree flex h-7 cursor-pointer items-center gap-1 rounded-md pr-2 text-muted-foreground outline-none select-none",
              "hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/60",
              "aria-selected:bg-accent aria-selected:font-medium aria-selected:text-foreground",
            )}
            style={{ paddingLeft: `${0.25 + Math.max(0, row.depth - 1) * 0.875}rem` }}
          >
            {row.hasChildren ? (
              <span
                role="presentation"
                onClick={(event) => {
                  event.stopPropagation()
                  toggle(row.path)
                }}
                className="grid size-5 shrink-0 place-items-center rounded hover:bg-background"
              >
                <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} aria-hidden />
              </span>
            ) : (
              <span className={cn("shrink-0", row.depth === 0 ? "w-0.5" : "w-5")} aria-hidden />
            )}
            <Icon className={cn("size-3.5 shrink-0", isSelected && "text-primary")} aria-hidden />
            <span className="min-w-0 flex-1 truncate" title={row.path || undefined}>
              {row.name}
            </span>
            <span className="shrink-0 text-[0.7rem] text-muted-foreground tabular-nums">{row.count.toLocaleString()}</span>
          </li>
        )
      })}
    </ul>
  )
}
