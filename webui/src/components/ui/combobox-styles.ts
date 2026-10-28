/** Classes shared by the chips combobox (`combobox.tsx`) and the stand-in a form shows until it loads (`ui/inputs/MultiChoice.tsx`). */
export const COMBOBOX_CHIPS_CLASS =
  "flex min-h-8 flex-wrap items-center gap-1 rounded-sm border border-transparent bg-field shadow-[inset_0_-1px_0_var(--input)] px-2.5 py-1 text-sm transition-colors focus-within:shadow-[inset_0_-2px_0_var(--ring)] has-aria-invalid:border-destructive has-aria-invalid:ring-3 has-aria-invalid:ring-destructive/20 has-data-[slot=combobox-chip]:px-1 dark:has-aria-invalid:border-destructive/50 dark:has-aria-invalid:ring-destructive/40"

export const COMBOBOX_CHIP_CLASS =
  "flex h-[calc(--spacing(5.25))] w-fit items-center justify-center gap-1 rounded-sm bg-muted px-1.5 text-xs font-medium whitespace-nowrap text-foreground has-disabled:pointer-events-none has-disabled:cursor-not-allowed has-disabled:opacity-50 has-data-[slot=combobox-chip-remove]:pr-0"

export const COMBOBOX_CHIP_REMOVE_CLASS = "-ml-1 opacity-50 hover:opacity-100"

export const COMBOBOX_CHIPS_INPUT_CLASS = "min-w-16 flex-1 outline-none"
