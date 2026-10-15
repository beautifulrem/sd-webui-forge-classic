import { FileSearch } from "lucide-react"
import { PagePlaceholder } from "@/components/PagePlaceholder"

export default function PngInfoPage() {
  return (
    <PagePlaceholder
      icon={FileSearch}
      title="PNG Info"
      description="Read generation parameters from an image and send them to txt2img or img2img. This view is being rebuilt."
    />
  )
}
