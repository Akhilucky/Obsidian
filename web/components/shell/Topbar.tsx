"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import { Command, PanelLeftClose, PanelLeftOpen, Search } from "lucide-react"
import StatusDot from "@/components/ui/StatusDot"
import { api } from "@/lib/api"
import type { Health } from "@/lib/types"

type Props = {
  collapsed: boolean
  onToggle: () => void
  onCommand: () => void
}

export default function Topbar({ collapsed, onToggle, onCommand }: Props) {
  const [health, setHealth] = useState<Health | null>(null)

  useEffect(() => {
    let alive = true
    const tick = async () => {
      const h = await api.health()
      if (alive) setHealth(h)
    }
    tick()
    const id = setInterval(tick, 15000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  return (
    <motion.header
      initial={{ y: -12, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="fixed right-0 top-0 z-30 flex h-[64px] items-center justify-between border-b backdrop-blur-xl"
      style={{
        left: collapsed ? 80 : 264,
        borderColor: "var(--border)",
        background: "rgba(9,9,11,0.72)",
        transition: "left 260ms cubic-bezier(0.32,0.72,0,1)",
      }}
    >
      <div className="flex items-center gap-2 pl-5">
        <button
          onClick={onToggle}
          aria-label="Toggle sidebar"
          className="rounded-lg p-2 text-[var(--text-muted)] transition-all duration-150 hover:bg-[var(--hover)] hover:text-[var(--text-primary)] active:scale-95"
        >
          {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
        </button>
      </div>

      {/* Command palette trigger */}
      <button
        onClick={onCommand}
        className="group mr-5 flex items-center gap-2.5 rounded-xl border px-3.5 py-2 transition-all duration-180 hover:border-[var(--border-strong)]"
        style={{ borderColor: "var(--border)", background: "rgba(255,255,255,0.02)" }}
      >
        <Search size={14} className="text-[var(--text-muted)] transition-colors group-hover:text-[var(--accent)]" />
        <span className="text-[12px] text-[var(--text-muted)]">Search…</span>
        <kbd className="rounded-md border px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-muted)]"
          style={{ borderColor: "var(--border-strong)" }}
        >
          <Command size={9} className="inline" /> K
        </kbd>
      </button>

      {/* Right status cluster */}
      <div className="mr-6 hidden items-center gap-5 md:flex">
        <div className="flex items-center gap-2 font-mono text-[11px] text-[var(--text-muted)]">
          <StatusDot status={health?.market_open ? "up" : "neutral"} size={5} />
          {health ? (health.market_open ? "Market Open" : "Market Closed") : "…"}
          <span className="text-[var(--border-strong)]">|</span>
          {health?.time ?? "—"}
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px] text-[var(--text-muted)]">
          <span
            className="h-5 w-5 rounded-md"
            style={{
              background: "linear-gradient(135deg,#38bdf8,#2f6fed)",
              boxShadow: "0 0 12px rgba(56,189,248,0.3)",
            }}
          />
          Terminal
        </div>
      </div>
    </motion.header>
  )
}
