/**
 * The live preview, double-buffered like the Gradio UI's: each new frame
 * (already decoded, see api/ws.ts) is stacked over the previous one, so the
 * stage never blinks empty between steps. At most two frames are mounted; the
 * one underneath is decorative. No fade between them: a step is one drawn
 * frame (docs/remi-webui/frontend.md, "Motion"); a 150 ms fade-in made the
 * page draw about 12 frames per step and multiplied the browser's CPU per step
 * (docs/research/ui-performance.md, "Per-step cost of the page").
 */
import { useState } from "react"
import { cn } from "@/lib/utils"

type Frame = { src: string; key: number }

export function LivePreview({ src, alt, className }: { src: string; alt: string; className?: string }) {
  const [frames, setFrames] = useState<Frame[]>(() => [{ src, key: 0 }])
  const top = frames.at(-1)
  // A new frame: keep the current one underneath the new one, so the stage never blinks.
  if (top && top.src !== src) setFrames([top, { src, key: top.key + 1 }])

  return (
    <span className={cn("relative block size-full", className)}>
      {frames.map((frame, index) => {
        const current = index === frames.length - 1
        return (
          <img
            key={frame.key}
            src={frame.src}
            alt={current ? alt : ""}
            aria-hidden={current ? undefined : true}
            draggable={false}
            data-frame={current ? "current" : "previous"}
            // Previews are often latent-sized (e.g. 96 px): scale them up to the stage.
            className="absolute inset-0 size-full object-contain"
          />
        )
      })}
    </span>
  )
}
