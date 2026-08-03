"use client"

import { useMemo, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Check, ChevronDown, Search } from "lucide-react"
import { useWatchlist } from "@/lib/watchlist-store"

const SUGGESTIONS = [
  "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "JPM", "V",
  "NFLX", "ORCL", "CRM", "COST", "LLY", "BA", "RELIANCE.NS", "TCS.NS",
  "HDFCBANK.NS", "INFY.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
]

type Props = {
  ticker: string
  onChange: (ticker: string) => void
}

export default function FocusSelector({ ticker, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")
  const pin = useWatchlist((s) => s.pin)

  const matches = useMemo(() => {
    const q = query.trim().toUpperCase()
    if (!q) return SUGGESTIONS
    return [...new Set([
      ...SUGGESTIONS.filter((s) => s.includes(q)),
      ...(/^[A-Z0-9.\^]{1,12}$/.test(q) ? [q] : []),
    ])].slice(0, 12)
  }, [query])

  const select = (t: string) => {
    onChange(t)
    setOpen(false)
    setQuery("")
    pin(t)
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-xl border px-3 py-1.5 font-mono text-[13px] font-semibold transition-all duration-150 hover:border-[var(--border-strong)] active:scale-95"
        style={{ borderColor: "var(--border)", background: "rgba(255,255,255,0.02)", color: "var(--accent)" }}
      >
        {ticker}
        <ChevronDown size={13} className={`transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
      </button>
      <AnimatePresence>
        {open && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
            <motion.div
              initial={{ opacity: 0, y: -6, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -4, scale: 0.98 }}
              transition={{ duration: 0.16, ease: "easeOut" }}
              className="absolute left-0 z-50 mt-2 w-64 overflow-hidden rounded-xl border backdrop-blur-2xl"
              style={{
                borderColor: "var(--border-strong)",
                background: "rgba(13,13,18,0.95)",
                boxShadow: "0 16px 48px rgba(0,0,0,0.55)",
              }}
            >
              <div className="flex items-center gap-2 border-b px-3 py-2" style={{ borderColor: "var(--border)" }}>
                <Search size={12} className="text-[var(--text-muted)]" />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search or enter ticker…"
                  autoFocus
                  className="w-full bg-transparent text-[12px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none"
                />
              </div>
              <div className="max-h-64 overflow-y-auto p-1.5">
                {matches.map((t) => (
                  <button
                    key={t}
                    onClick={() => select(t)}
                    className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-left font-mono text-[12px] transition-colors hover:bg-[var(--hover)]"
                    style={{ color: t === ticker ? "var(--accent)" : "var(--text-secondary)" }}
                  >
                    {t}
                    {t === ticker && <Check size={13} />}
                  </button>
                ))}
                {matches.length === 0 && (
                  <div className="px-3 py-6 text-center text-[11px] text-[var(--text-muted)]">
                    No matches for “{query}”
                  </div>
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
