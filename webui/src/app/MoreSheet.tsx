/** Mobile "More" sheet: the destinations that do not fit the bottom bar. */
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { BottomTab } from "./NavRail"
import { isActive, type NavItem } from "./routes"
import type { Route } from "./router"

export default function MoreSheet({ items, route, onClose }: { items: NavItem[]; route: Route; onClose: () => void }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(true)
  const close = () => setOpen(false)
  return (
    <Sheet open={open} onOpenChange={(next) => !next && close()} onOpenChangeComplete={(next) => !next && onClose()}>
      <SheetContent side="bottom" className="pb-[max(1rem,env(safe-area-inset-bottom))]">
        <SheetHeader>
          <SheetTitle>{t("More")}</SheetTitle>
        </SheetHeader>
        <div className="grid grid-cols-4 gap-2 px-4">
          {items.map((item) => (
            <BottomTab key={item.id} item={item} active={isActive(item, route)} onClick={close} />
          ))}
        </div>
        <Button variant="ghost" className="mx-4" onClick={close}>
          {t("Close")}
        </Button>
      </SheetContent>
    </Sheet>
  )
}
