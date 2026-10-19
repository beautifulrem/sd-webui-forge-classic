/** Markdown and sanitised HTML display components (logic in rich-text.tsx). */
import { renderMarkdown, sanitizeHtml } from "./rich-text"

export function Markdown({ text, className }: { text: string; className?: string }) {
  return <div className={className ?? "space-y-2 text-sm leading-relaxed text-foreground/90"}>{renderMarkdown(text)}</div>
}

export function SafeHtml({ html, className }: { html: string; className?: string }) {
  return (
    <div
      className={
        className ??
        "space-y-2 text-sm leading-relaxed [&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2 [&_a:hover]:decoration-2 [&_code]:font-mono [&_table]:w-full [&_td]:border [&_td]:px-2 [&_th]:border [&_th]:px-2"
      }
    >
      {sanitizeHtml(html)}
    </div>
  )
}
