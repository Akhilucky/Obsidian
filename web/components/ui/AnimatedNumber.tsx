"use client"

import { useEffect, useRef, useState } from "react"

type Props = {
  value: number
  format?: (v: number) => string
  duration?: number
  className?: string
}

export default function AnimatedNumber({
  value,
  format,
  duration = 600,
  className,
}: Props) {
  const [display, setDisplay] = useState(value)
  const prevRef = useRef(value)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    const from = prevRef.current
    const to = value
    if (from === to) return
    const start = performance.now()

    const step = (now: number) => {
      const t = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - t, 3)
      setDisplay(from + (to - from) * eased)
      if (t < 1) {
        rafRef.current = requestAnimationFrame(step)
      } else {
        prevRef.current = to
      }
    }

    rafRef.current = requestAnimationFrame(step)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      prevRef.current = to
    }
  }, [value, duration])

  const fmt = format ?? ((v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 2 }))
  return <span className={className}>{fmt(display)}</span>
}
