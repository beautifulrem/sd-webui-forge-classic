/**
 * One menu for every card: at the pointer for a right click, under the "…"
 * button, or at the card for Shift+F10 / the Menu key on the grid. Edit, set
 * preview and open folder are offered for items with a model file only.
 */
import { Menu } from "@base-ui/react/menu"
import { Copy, FolderOpen, ImagePlus, Info, MinusCircle, Pencil, PlusCircle, Power } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { NetworkItem, NetworkKind } from "@/api/types"
import { DropdownMenuItem, DropdownMenuSeparator } from "@/components/ui/dropdown-menu"
import { isEditable, type MenuAnchor } from "./items"

export type MenuCommand = "apply" | "metadata" | "edit" | "preview" | "copy" | "folder"

/** A pointer position as a zero-size anchor (Floating UI's virtual element). */
function positionerAnchor(anchor: MenuAnchor | null) {
  if (!anchor || anchor instanceof HTMLElement) return anchor
  const { x, y, element } = anchor
  return { getBoundingClientRect: () => ({ x, y, left: x, top: y, right: x, bottom: y, width: 0, height: 0 }), contextElement: element }
}

export function ItemMenu({
  item,
  kind,
  anchor,
  active,
  canOpenFolder,
  onClose,
  onCommand,
}: {
  item: NetworkItem | null
  kind: NetworkKind | undefined
  anchor: MenuAnchor | null
  active: boolean
  canOpenFolder: boolean
  onClose: () => void
  onCommand: (command: MenuCommand, item: NetworkItem) => void
}) {
  const { t } = useTranslation()
  const open = Boolean(item && anchor)
  const atPointer = Boolean(anchor && !(anchor instanceof HTMLElement))
  const editable = item ? isEditable(item) : false
  const run = (command: MenuCommand) => () => {
    if (item) onCommand(command, item)
  }
  const applyLabel =
    kind?.action === "checkpoint"
      ? t("Load this model")
      : kind?.action === "module"
        ? active
          ? t("Remove from the modules")
          : t("Add to the modules")
        : active
          ? t("Remove from the prompt")
          : t("Add to the prompt")
  const ApplyIcon = kind?.action === "checkpoint" ? Power : active ? MinusCircle : PlusCircle
  return (
    <Menu.Root
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose()
      }}
    >
      <Menu.Portal>
        <Menu.Positioner anchor={positionerAnchor(anchor)} side="bottom" align={atPointer ? "start" : "end"} sideOffset={atPointer ? 2 : 4} className="isolate z-50 outline-none">
          <Menu.Popup
            aria-label={item?.title}
            className="z-50 max-h-(--available-height) min-w-48 origin-(--transform-origin) overflow-y-auto rounded-lg bg-popover p-1 text-popover-foreground shadow-md ring-1 ring-foreground/10 duration-100 outline-none data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95"
          >
            {kind && kind.action !== "none" && !(kind.action === "checkpoint" && active) ? (
              <DropdownMenuItem onClick={run("apply")}>
                <ApplyIcon aria-hidden />
                {applyLabel}
              </DropdownMenuItem>
            ) : null}
            <DropdownMenuItem onClick={run("metadata")}>
              <Info aria-hidden />
              {t("Metadata")}
              <kbd className="ml-auto text-xs text-muted-foreground">I</kbd>
            </DropdownMenuItem>
            {editable && (kind?.fields.length ?? 0) > 0 ? (
              <DropdownMenuItem onClick={run("edit")}>
                <Pencil aria-hidden />
                {t("Edit metadata")}
                <kbd className="ml-auto text-xs text-muted-foreground">E</kbd>
              </DropdownMenuItem>
            ) : null}
            {editable ? (
              <DropdownMenuItem onClick={run("preview")}>
                <ImagePlus aria-hidden />
                {t("Set preview")}
              </DropdownMenuItem>
            ) : null}
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={run("copy")}>
              <Copy aria-hidden />
              {t("Copy path")}
            </DropdownMenuItem>
            {canOpenFolder && editable ? (
              <DropdownMenuItem onClick={run("folder")}>
                <FolderOpen aria-hidden />
                {t("Open folder")}
              </DropdownMenuItem>
            ) : null}
          </Menu.Popup>
        </Menu.Positioner>
      </Menu.Portal>
    </Menu.Root>
  )
}
