/**
 * Extension pages (former `on_ui_tabs`). A page with `widget` renders the
 * dedicated view `src/pages/custom/<widget>/index.tsx`; otherwise its layout
 * comes from GET /api/v2/pages/{id} and goes through the schema renderer.
 */
import { useQuery, useSuspenseQuery } from "@tanstack/react-query"
import { Puzzle } from "lucide-react"
import { type ComponentType, lazy, type LazyExoticComponent, Suspense } from "react"
import { bootstrapQuery, pageSchemaQuery } from "@/api/queries"
import type { PageInfo } from "@/api/types"
import type { Route } from "@/app/router"
import { PagePlaceholder } from "@/components/PagePlaceholder"
import { PageSkeleton } from "@/components/PageSkeleton"
import { PageForm } from "@/ui/SchemaForm"

export type CustomPageProps = { page: PageInfo }

const customModules = import.meta.glob<{ default: ComponentType<CustomPageProps> }>("../custom/*/index.tsx")
const customPages = new Map<string, LazyExoticComponent<ComponentType<CustomPageProps>>>(
  Object.entries(customModules).map(([path, load]) => [path.split("/").at(-2) ?? path, lazy(load)]),
)

function SchemaPage({ page }: { page: PageInfo }) {
  const { data } = useSuspenseQuery(pageSchemaQuery(page.id))
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl p-4">
        <PageForm schema={data} />
      </div>
    </div>
  )
}

export default function ExtensionPage({ route }: { route: Extract<Route, { name: "page" }> }) {
  const bootstrap = useQuery(bootstrapQuery)
  if (bootstrap.isPending) return <PageSkeleton />
  const page = bootstrap.data?.pages.find((p) => p.id === route.id)
  if (!page) {
    return <PagePlaceholder icon={Puzzle} title="Page not found" description="No extension provides this page." />
  }
  const Custom = page.widget ? customPages.get(page.widget) : undefined
  if (page.widget && !Custom) {
    return <PagePlaceholder icon={Puzzle} title={page.title} description="This page's view is not part of this build." />
  }
  return (
    <Suspense fallback={<PageSkeleton />}>
      {/* oxlint-disable-next-line react/static-components -- a stable lazy component from the page registry */}
      {Custom ? <Custom page={page} /> : <SchemaPage page={page} />}
    </Suspense>
  )
}
