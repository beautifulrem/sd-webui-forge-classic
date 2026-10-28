/** "Send to <page>" buttons for an image (PNG Info); nothing when no page accepts images. See `./targets`. */
import { Send } from "@/components/icons"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { sendImageToPage, usePageSendTargets } from "./targets"

export function PageSendButtons({ fileId, size = "sm" }: { fileId: string; size?: "sm" | "default" }) {
  const { t } = useTranslation()
  const targets = usePageSendTargets("image")
  if (targets.length === 0) return null
  return (
    <>
      {targets.map((item) => (
        <Button key={`${item.pageId}:${item.target.param}`} size={size} variant="outline" onClick={() => void sendImageToPage(item, fileId)}>
          <Send aria-hidden />
          {t(item.target.label)}
        </Button>
      ))}
    </>
  )
}
