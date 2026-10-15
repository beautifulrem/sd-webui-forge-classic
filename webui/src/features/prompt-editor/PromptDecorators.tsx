/** Renders the registered prompt decorators around an editor (see `decorators.ts`). */
import { useSyncExternalStore } from "react"
import { type PromptDecoratorProps, promptDecorators, subscribePromptDecorators } from "./decorators"

/** The editor wrapped in every registered decorator (just the editor when none is). */
export function PromptDecorators(props: PromptDecoratorProps) {
  const decorators = useSyncExternalStore(subscribePromptDecorators, promptDecorators, promptDecorators)
  let node = props.children
  for (const { id, component: Decorator } of decorators) {
    node = (
      <Decorator key={id} {...props}>
        {node}
      </Decorator>
    )
  }
  return <>{node}</>
}
