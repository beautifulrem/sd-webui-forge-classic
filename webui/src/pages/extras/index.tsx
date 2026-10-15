import { ImageUpscale } from "lucide-react"
import { PagePlaceholder } from "@/components/PagePlaceholder"

export default function ExtrasPage() {
  return (
    <PagePlaceholder
      icon={ImageUpscale}
      title="Extras"
      description="Upscaling and post-processing of single images and batches. This view is being rebuilt."
    />
  )
}
