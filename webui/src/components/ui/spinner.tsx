import { cn } from "cn"

import { Loader2Icon } from "@/components/icons"
import { useTranslation } from "react-i18next"

function Spinner({ className, ...props }: React.ComponentProps<"svg">) {
  const { t } = useTranslation()
  return (
    <Loader2Icon
      data-slot="spinner"
      role="status"
      aria-label={t("Loading")}
      className={cn("size-4 animate-spin", className)}
      {...props}
    />
  )
}

export { Spinner }
