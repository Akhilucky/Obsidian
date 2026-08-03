"use client"

import { motion } from "framer-motion"
import { Pin, PinOff } from "lucide-react"
import { useWatchlist } from "@/lib/watchlist-store"

type Props = {
  symbol: string
  size?: number
  className?: string
}

export default function PinButton({ symbol, size = 15, className = "" }: Props) {
  const isPinned = useWatchlist((s) => s.isPinned(symbol))
  const pin = useWatchlist((s) => s.pin)
  const unpin = useWatchlist((s) => s.unpin)

  return (
    <motion.button
      whileTap={{ scale: 0.85 }}
      aria-label={isPinned ? `Unpin ${symbol}` : `Pin ${symbol}`}
      title={isPinned ? "Unpin from watchlist" : "Pin to watchlist"}
      onClick={(e) => {
        e.stopPropagation()
        e.preventDefault()
        if (isPinned) unpin(symbol)
        else pin(symbol)
      }}
      className={`inline-flex items-center justify-center rounded-lg p-1.5 transition-all duration-150 ${
        isPinned
          ? "text-[var(--accent)]"
          : "text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text-secondary)]"
      } ${className}`}
      style={
        isPinned
          ? { background: "rgba(56,189,248,0.08)", boxShadow: "0 0 10px rgba(56,189,248,0.15)" }
          : undefined
      }
    >
      {isPinned ? <Pin size={size} fill="currentColor" /> : <PinOff size={size} />}
    </motion.button>
  )
}
