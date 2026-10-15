/**
 * Icons extension pages may name in `Page(icon=...)`. A curated map keeps the
 * shell bundle small (no dynamic import of all of lucide); unknown names fall
 * back to a puzzle piece. Add entries here when a built-in page needs one.
 */
import {
  BookOpen,
  Boxes,
  Brush,
  CalendarClock,
  Camera,
  ChartLine,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  GitMerge,
  Grid3x3,
  Image,
  Layers,
  ListChecks,
  type LucideIcon,
  Palette,
  Puzzle,
  ScrollText,
  Sparkles,
  SlidersHorizontal,
  Tags,
  Terminal,
  Wand2,
  Wrench,
} from "lucide-react"

const PAGE_ICONS: Record<string, LucideIcon> = {
  "book-open": BookOpen,
  boxes: Boxes,
  brush: Brush,
  "calendar-clock": CalendarClock,
  camera: Camera,
  "chart-line": ChartLine,
  cpu: Cpu,
  database: Database,
  "flask-conical": FlaskConical,
  gauge: Gauge,
  "git-merge": GitMerge,
  grid: Grid3x3,
  image: Image,
  layers: Layers,
  "list-checks": ListChecks,
  palette: Palette,
  puzzle: Puzzle,
  "scroll-text": ScrollText,
  sparkles: Sparkles,
  sliders: SlidersHorizontal,
  tags: Tags,
  terminal: Terminal,
  wand: Wand2,
  wrench: Wrench,
}

export function pageIcon(name: string | null | undefined): LucideIcon {
  return (name && PAGE_ICONS[name]) || Puzzle
}
