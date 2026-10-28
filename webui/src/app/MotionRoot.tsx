/**
 * Motion (and the icon defaults) for the whole app: `m` components render at once and start
 * animating when the feature bundle arrives (its own chunk, off the first
 * load). Reduced-motion users get no transform or layout animation.
 */
import { LazyMotion, MotionConfig } from "motion/react"
import type { ReactNode } from "react"
import { IconDefaults } from "@/components/icons"

const features = () => import("@/lib/motion-features").then((module) => module.default)

export function MotionRoot({ children }: { children: ReactNode }) {
  return (
    <LazyMotion features={features} strict>
      <MotionConfig reducedMotion="user" transition={{ type: "spring", visualDuration: 0.22, bounce: 0 }}>
        <IconDefaults>{children}</IconDefaults>
      </MotionConfig>
    </LazyMotion>
  )
}
