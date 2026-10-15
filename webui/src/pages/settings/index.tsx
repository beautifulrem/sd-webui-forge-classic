import { Settings } from "lucide-react"
import { PagePlaceholder } from "@/components/PagePlaceholder"

export default function SettingsPage() {
  return (
    <PagePlaceholder
      icon={Settings}
      title="Settings"
      description="All options, searchable, rendered from the settings schema. This view is being rebuilt."
    />
  )
}
