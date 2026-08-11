import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cx } from '@/lib/format'

type Variant = 'primary' | 'ghost' | 'outline' | 'danger'
type Size = 'sm' | 'md' | 'icon'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  children?: ReactNode
}

const VARIANTS: Record<Variant, string> = {
  primary:
    'bg-accent text-canvas hover:brightness-110 active:brightness-95 disabled:hover:brightness-100 font-medium',
  ghost: 'text-muted hover:bg-raised hover:text-ink',
  outline: 'border border-hairline bg-surface text-ink hover:border-accent/50 hover:bg-raised',
  danger: 'text-danger hover:bg-danger/10',
}

const SIZES: Record<Size, string> = {
  sm: 'h-7 gap-1.5 px-2.5 text-xs',
  md: 'h-9 gap-2 px-3.5 text-sm',
  icon: 'h-7 w-7 justify-center',
}

export function Button({ variant = 'ghost', size = 'md', className, ...props }: Props) {
  return (
    <button
      type="button"
      className={cx(
        'inline-flex select-none items-center rounded-md transition-colors',
        'disabled:cursor-not-allowed disabled:opacity-45',
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...props}
    />
  )
}
