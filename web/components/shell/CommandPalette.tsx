"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import { AnimatePresence, motion } from "framer-motion"
import { BarChart3, Bot, FlaskConical, Landmark, LayoutDashboard, Radar, Settings, Wallet } from "lucide-react"

const ITEMS = [
  { label: "Overview", path: "/overview", icon: LayoutDashboard, keywords: "home dashboard market" },
  { label: "Analysis", path: "/analysis", icon: BarChart3, keywords: "chart ticker stock technical" },
  { label: "Signals", path: "/signals", icon: Radar, keywords: "alerts buy sell alpha" },
  { label: "India", path: "/india", icon: Landmark, keywords: "nse bse nifty sensex" },
  { label: "Portfolio", path: "/portfolio", icon: Wallet, keywords: "holdings pnl assets" },
  { label: "Research", path: "/research", icon: FlaskConical, keywords: "backtest factor models" },
  { label: "Agents", path: "/agents", icon: Bot, keywords: "pipeline orchestrator ai" },
  { label: "Settings", path: "/settings", icon: Settings, keywords: "config cache system" },
]

type Props = {
  open: boolean
  onClose: () => void
}

export default function CommandPalette({ open, onClose }: Props) {
  const router = useRouter()
  const [query, setQuery] = useState("")
  const [index, setIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  const results = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return ITEMS
    return ITEMS.filter(
      (i) =>
        i.label.toLowerCase().includes(q) ||
        i.path.includes(q) ||
        i.keywords.includes(q)
    )
  }, [query])

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 40)
    }
  }, [open])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setQuery("")
        setIndex(0)
        onClose()
      }
    }
    if (open) window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  const close = () => {
    setQuery("")
    setIndex(0)
    onClose()
  }

  const go = (path: string) => {
    setQuery("")
    setIndex(0)
    onClose()
    router.push(path)
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
            
onClick={close}
          />
          <motion.div
            initial={{ opacity: 0, y: -14, scale: 0.985, filter: "blur(6px)" }}
            animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -10, scale: 0.99, filter: "blur(4px)" }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="fixed left-1/2 top-[18%] z-50 w-[520px] max-w-[92vw] -translate-x-1/2 overflow-hidden rounded-2xl border backdrop-blur-2xl"
            style={{
              borderColor: "var(--border-strong)",
              background: "rgba(13,13,18,0.94)",
              boxShadow: "0 24px 80px rgba(0,0,0,0.6)",
            }}
          >
            <div className="flex items-center gap-3 border-b px-4 py-3.5" style={{ borderColor: "var(--border)" }}>
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value)
                  setIndex(0)
                }}
                onKeyDown={(e) => {
                  if (e.key === "ArrowDown") {
                    e.preventDefault()
                    setIndex((i) => Math.min(i + 1, results.length - 1))
                  } else if (e.key === "ArrowUp") {
                    e.preventDefault()
                    setIndex((i) => Math.max(i - 1, 0))
                  } else if (e.key === "Enter" && results[index]) {
                    go(results[index].path)
                  }
                }}
                placeholder="Jump to a workspace…"
                className="flex-1 bg-transparent text-[14px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none"
              />
              <kbd className="rounded-md border px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-muted)]"
                style={{ borderColor: "var(--border-strong)" }}
              >
                ESC
              </kbd>
            </div>
            <div className="max-h-[320px] overflow-y-auto p-2">
              {results.length === 0 && (
                <div className="px-4 py-8 text-center text-[13px] text-[var(--text-muted)]">
                  No results for “{query}”
                </div>
              )}
              {results.map((item, i) => {
                const Icon = item.icon
                return (
                  <button
                    key={item.path}
                    onClick={() => go(item.path)}
                    onMouseEnter={() => setIndex(i)}
                    className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors duration-120"
                    style={{
                      background: i === index ? "rgba(56,189,248,0.10)" : "transparent",
                    }}
                  >
                    <Icon
                      size={16}
                      strokeWidth={1.8}
                      style={{ color: i === index ? "var(--accent)" : "var(--text-muted)" }}
                    />
                    <span
                      className="text-[13px]"
                      style={{ color: i === index ? "var(--text-primary)" : "var(--text-secondary)" }}
                    >
                      {item.label}
                    </span>
                    <span className="ml-auto font-mono text-[10px] text-[var(--text-muted)]">{item.path}</span>
                  </button>
                )
              })}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
