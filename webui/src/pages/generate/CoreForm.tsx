/**
 * The curated core form in the schema's (the user's) category order: size
 * blocks get aspect presets and "W x H" paste, guidance blocks a card, and
 * sampler, seed, sections (Hires. fix, refiner, …) and anything else render
 * as the schema renderer draws them.
 */
import type { ReactNode } from "react"
import { useId } from "react"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { Img2imgSourcePanel } from "@/canvas"
import { OwnerProvider } from "@/ui/context"
import { RenderNode } from "@/ui/Renderer"
import type { Node } from "@/ui/schema"
import type { BlockKind, CuratedForm } from "./curate"
import { usePasteSize } from "./size-hooks"
import { SizeTools } from "./SizeTools"

export function Card({ title, actions, children, className, onPaste }: { title: string; actions?: ReactNode; children: ReactNode; className?: string; onPaste?: (event: React.ClipboardEvent) => void }) {
  const id = useId()
  return (
    <section aria-labelledby={id} onPaste={onPaste} className={cn("flex min-w-0 flex-col gap-3 rounded-md bg-card p-3 text-card-foreground", className)}>
      <div className="flex min-h-7 items-center gap-2">
        <h2 id={id} className="text-[1.0625rem] font-semibold tracking-[-0.01em]">
          {title}
        </h2>
        {actions ? <div className="ml-auto flex items-center gap-1">{actions}</div> : null}
      </div>
      {children}
    </section>
  )
}

function Blocks({ nodes }: { nodes: readonly Node[] }) {
  return nodes.map((node, index) => <RenderNode key={index} node={node} />)
}

/** Consecutive blocks of one kind, in schema order. */
function runs(blocks: CuratedForm["blocks"]): { kind: BlockKind; nodes: Node[] }[] {
  const out: { kind: BlockKind; nodes: Node[] }[] = []
  for (const block of blocks) {
    const last = out.at(-1)
    if (last && last.kind === block.kind) last.nodes.push(block.node)
    else out.push({ kind: block.kind, nodes: [block.node] })
  }
  return out
}

export function CoreForm({ curated }: { curated: CuratedForm }) {
  const { t } = useTranslation()
  const pasteSize = usePasteSize()
  return (
    <OwnerProvider owner="core">
      <div className="flex flex-col gap-3">
        {runs(curated.blocks.filter((block) => block.kind !== "source")).map(({ kind, nodes }, index) => {
          switch (kind) {
            case "size":
              return (
                <Card key={index} title={t("Size")} actions={<SizeTools />} onPaste={pasteSize}>
                  <Blocks nodes={nodes} />
                </Card>
              )
            case "guidance":
              return (
                <Card key={index} title={t("Guidance")}>
                  <Blocks nodes={nodes} />
                </Card>
              )
            default:
              // Slots frame their scripts; sections are panels already.
              return (
                <div key={index} className="flex min-w-0 flex-col gap-3">
                  <Blocks nodes={nodes} />
                </div>
              )
          }
        })}
      </div>
    </OwnerProvider>
  )
}

/**
 * img2img source: the mode picker, the source image and mask lines and the
 * mode's options (`canvas/img2img/SourcePanel`); the image is edited on the
 * stage.
 */
export function SourceSlot({ curated }: { curated: CuratedForm }) {
  const { t } = useTranslation()
  const nodes = curated.blocks.filter((block) => block.kind === "source").map((block) => block.node)
  if (nodes.length === 0) return null
  return (
    <Card title={t("Source")}>
      <div data-slot="img2img-source">
        <Img2imgSourcePanel nodes={nodes} />
      </div>
    </Card>
  )
}
