import { Loader2 } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { useInheritedEnabled, useRunAction } from "../hooks"
import type { ButtonNode } from "../schema"

const VARIANT = {
  primary: { variant: "default", size: "default" },
  secondary: { variant: "secondary", size: "default" },
  ghost: { variant: "ghost", size: "default" },
  tool: { variant: "outline", size: "sm" },
  danger: { variant: "destructive", size: "default" },
} as const

/** Calls the owner's action with the owner's current values. */
export function ActionButton({ node }: { node: ButtonNode }) {
  const { t } = useTranslation()
  const run = useRunAction()
  const enabled = useInheritedEnabled()
  const [busy, setBusy] = useState(false)
  const style = VARIANT[node.variant ?? "secondary"] ?? VARIANT.secondary

  return (
    <Button
      variant={style.variant}
      size={style.size}
      disabled={!enabled || busy}
      aria-busy={busy || undefined}
      onClick={async () => {
        setBusy(true)
        try {
          await run(node.action)
        } finally {
          setBusy(false)
        }
      }}
      className="self-start"
    >
      {busy ? <Loader2 className="animate-spin" aria-hidden /> : null}
      {t(node.label)}
    </Button>
  )
}
