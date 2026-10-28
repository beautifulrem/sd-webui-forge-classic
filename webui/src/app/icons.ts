/**
 * Icons extension pages may name in `Page(icon=...)` (Lucide's kebab-case
 * names, the API's vocabulary; drawn from the app's own set). A curated map
 * keeps the shell bundle small; unknown names fall back to a puzzle piece. Add entries here when a built-in page needs one.
 */
import {
  BookOpen,
  Boxes,
  Braces,
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
  type IconComponent,
  Palette,
  Puzzle,
  ScrollText,
  Sparkles,
  SlidersHorizontal,
  Tags,
  Terminal,
  Wand2,
  Wrench,
} from "@/components/icons"

const PAGE_ICONS: Record<string, IconComponent> = {
  "book-open": BookOpen,
  boxes: Boxes,
  braces: Braces,
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

export function pageIcon(name: string | null | undefined): IconComponent {
  return (name && PAGE_ICONS[name]) || Puzzle
}
