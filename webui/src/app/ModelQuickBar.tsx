/**
 * Header model quick-bar: the active checkpoint, switchable in place. The full
 * quick-bar (VAE / text encoders / presets) builds on this in a later wave.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { Box, Loader2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { useProgress } from "@/api/progress"
import { bootstrapQuery, keys, modelsQuery, queryClient } from "@/api/queries"
import { SearchSelect } from "@/components/SearchSelect"

const CHECKPOINT_OPTION = "sd_model_checkpoint"

export default function ModelQuickBar() {
  const { t } = useTranslation()
  const bootstrap = useQuery(bootstrapQuery)
  const models = useQuery(modelsQuery)
  const loading = useProgress((s) => s.modelLoading)
  const current =
    (bootstrap.data?.options[CHECKPOINT_OPTION] as string | undefined) ?? bootstrap.data?.models.checkpoint ?? null

  const switchModel = useMutation({
    mutationFn: (name: string) => api.patchOptions({ [CHECKPOINT_OPTION]: name }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: keys.app })
    },
    onError: (error) => toast.error(t("Could not switch model"), { description: error.message }),
  })

  const options = (models.data?.checkpoints ?? []).map((m) => ({ value: m.title ?? m.name, label: m.name }))
  const busy = switchModel.isPending || loading !== null

  return (
    <div className="flex min-w-0 items-center gap-2">
      <SearchSelect
        aria-label={t("Checkpoint")}
        size="sm"
        className="w-[min(22rem,40vw)]"
        options={options}
        value={switchModel.isPending ? switchModel.variables : current}
        placeholder={models.isPending ? t("Loading…") : t("Select a checkpoint")}
        onValueChange={(name) => switchModel.mutate(name)}
        disabled={busy}
        prefix={
          busy ? (
            <Loader2 className="size-3.5 shrink-0 animate-spin text-primary" aria-hidden />
          ) : (
            <Box className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
          )
        }
      />
      {loading ? (
        <span className="hidden truncate text-xs text-muted-foreground lg:inline" role="status">
          {t(loading.stage)}
        </span>
      ) : null}
    </div>
  )
}
