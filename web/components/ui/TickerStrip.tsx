"use client"

import { motion } from "framer-motion"
import AnimatedNumber from "./AnimatedNumber"
import StatusDot from "./StatusDot"
import type { IndexQuote } from "@/lib/types"

type Props = {
  items: Record<string, IndexQuote>
  loading?: boolean
}

export default function TickerStrip({ items, loading }: Props) {
  const entries = Object.entries(items)

  return (
    <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      {loading
        ? Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="obs-card p-3.5">
              <div className="skeleton h-3 w-16" />
              <div className="skeleton mt-2 h-5 w-20" />
            </div>
          ))
        : entries.map(([name, q], i) => {
            const up = (q?.change_pct ?? 0) >= 0
            return (
              <motion.div
                key={name}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, delay: i * 0.045, ease: "easeOut" }}
                className="obs-card obs-card-hover p-3.5"
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--text-muted)]">
                    {name}
                  </span>
                  <StatusDot status={up ? "up" : "down"} size={5} />
                </div>
                <div className="mt-1.5 font-mono text-[16px] font-semibold text-[var(--text-primary)]">
                  <AnimatedNumber
                    value={q?.price ?? 0}
                    format={(v) =>
                      v >= 1000
                        ? v.toLocaleString("en-US", { maximumFractionDigits: 0 })
                        : v.toFixed(2)
                    }
                  />
                </div>
                <div
                  className="mt-0.5 font-mono text-[11px]"
                  style={{ color: up ? "var(--up)" : "var(--down)" }}
                >
                  {q ? `${q.change_pct >= 0 ? "+" : ""}${q.change_pct.toFixed(2)}%` : "—"}
                </div>
              </motion.div>
            )
          })}
    </div>
  )
}
