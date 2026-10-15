/**
 * Boot-time registration of the prompt tools (imported once by main.tsx;
 * small: the completion engine, the tools bar and the panels are lazy).
 *
 * - tag autocompletion in every prompt editor (`registerCompletionSource`);
 * - the tools bar and tag-chips view around every prompt editor
 *   (`registerPromptDecorator`);
 * - fresh data when settings, models, networks or the history change.
 */
import { queryClient } from "@/api/queries"
import { onWsEvent } from "@/api/ws"
import { registerCompletionSource } from "@/features/prompt-editor/completion-registry"
import { registerPromptDecorator } from "@/features/prompt-editor/decorators"
import { PromptToolsDecorator } from "./PromptToolsDecorator"
import { toolKeys } from "./queries"

type Engine = typeof import("./completion/source")
let engine: Promise<Engine> | null = null
const loadEngine = () => (engine ??= import("./completion/source"))

function forgetCompletions() {
  if (engine) void engine.then((module) => module.clearCompletionCache())
}

registerCompletionSource("prompt-tools", async (context) => (await loadEngine()).promptToolsCompletions(context), { priority: 10 })
registerPromptDecorator("prompt-tools", PromptToolsDecorator)

onWsEvent("options.changed", () => {
  forgetCompletions()
  void queryClient.invalidateQueries({ queryKey: toolKeys.config })
  void queryClient.invalidateQueries({ queryKey: ["prompt-tools", "lookup"] })
})
onWsEvent("model.loaded", () => void queryClient.invalidateQueries({ queryKey: toolKeys.config }))
onWsEvent("networks.changed", forgetCompletions)
// the server records the prompts of a queued job a moment later
onWsEvent("job.queued", () => {
  setTimeout(() => void queryClient.invalidateQueries({ queryKey: ["prompt-tools", "history"] }), 500)
})
