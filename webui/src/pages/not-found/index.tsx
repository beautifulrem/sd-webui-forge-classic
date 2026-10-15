import { Compass } from "lucide-react"
import { useTranslation } from "react-i18next"
import { navigate } from "@/app/router"
import { PagePlaceholder } from "@/components/PagePlaceholder"
import { Button } from "@/components/ui/button"

export default function NotFoundPage() {
  const { t } = useTranslation()
  return (
    <PagePlaceholder icon={Compass} title="Page not found" description="This address does not match any page.">
      <Button variant="outline" onClick={() => navigate({ name: "generate", tab: "txt2img" })}>
        {t("Go to txt2img")}
      </Button>
    </PagePlaceholder>
  )
}
