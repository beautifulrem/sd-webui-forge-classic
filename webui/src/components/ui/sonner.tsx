"use client"

import { useTheme } from "@/lib/theme"
import { Toaster as Sonner, type ToasterProps } from "sonner"

import { CircleCheckIcon, InfoIcon, Loader2Icon, OctagonXIcon, TriangleAlertIcon } from "lucide-react"

/** `closeButtonAriaLabel`: the close buttons' label (sonner sets it per toast). */
const Toaster = ({ closeButtonAriaLabel, ...props }: ToasterProps & { closeButtonAriaLabel?: string }) => {
  const { theme = "system" } = useTheme()

  return (
    <Sonner
      theme={theme as ToasterProps["theme"]}
      className="toaster group"
      icons={{
        success: (
          <CircleCheckIcon
            className="size-4"
          />
        ),
        info: (
          <InfoIcon
            className="size-4"
          />
        ),
        warning: (
          <TriangleAlertIcon
            className="size-4"
          />
        ),
        error: (
          <OctagonXIcon
            className="size-4"
          />
        ),
        loading: (
          <Loader2Icon
            className="size-4 animate-spin"
          />
        ),
      }}
      style={
        {
          "--normal-bg": "var(--popover)",
          "--normal-text": "var(--popover-foreground)",
          "--normal-border": "var(--border)",
          "--border-radius": "var(--radius)",
        } as React.CSSProperties
      }
      toastOptions={{
        classNames: {
          toast: "cn-toast",
        },
        closeButtonAriaLabel,
      }}
      {...props}
    />
  )
}

export { Toaster }
