import { Skeleton } from "@/components/ui/skeleton"

/** Suspense fallback while a page chunk or its schema loads. */
export function PageSkeleton() {
  return (
    <div aria-busy="true" className="grid h-full gap-4 p-4 md:grid-cols-[minmax(20rem,28rem)_1fr]">
      <div className="space-y-3">
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-8 w-2/3" />
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-8 w-5/6" />
        <Skeleton className="h-32 w-full" />
      </div>
      <Skeleton className="hidden h-full min-h-64 w-full md:block" />
    </div>
  )
}
