import { TriangleAlert } from "@/components/icons"
import { Component, type ErrorInfo, type ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { t } from "@/i18n"

type Props = {
  children: ReactNode
  /** Changing this resets the boundary (e.g. the route key). */
  resetKey?: unknown
  /** Compact rendering for widgets inside a form. */
  inline?: boolean
}

type State = { error: Error | null; resetKey: unknown }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, resetKey: this.props.resetKey }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    return props.resetKey !== state.resetKey ? { error: null, resetKey: props.resetKey } : null
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    if (this.props.inline) {
      return (
        <div role="alert" className="rounded-lg border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
          {t("This control failed to render")}: {error.message}
        </div>
      )
    }
    return (
      <div role="alert" className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        <TriangleAlert className="size-8 text-destructive" aria-hidden />
        <div className="space-y-1">
          <p className="font-medium">{t("Something went wrong")}</p>
          <p className="max-w-md text-sm break-words text-muted-foreground">{error.message}</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => this.setState({ error: null })}>
          {t("Try again")}
        </Button>
      </div>
    )
  }
}
