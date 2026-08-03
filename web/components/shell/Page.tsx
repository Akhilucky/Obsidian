"use client"

import { motion } from "framer-motion"
import type { ReactNode } from "react"
import StatusDot from "@/components/ui/StatusDot"

type Badge = { label: string; tone?: "live" | "up" | "down" | "neutral" }

type Props = {
  title: string
  subtitle?: string
  badges?: Badge[]
  children: ReactNode
}

export default function Page({ title, subtitle, badges, children }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10, filter: "blur(8px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      exit={{ opacity: 0, y: -6, filter: "blur(10px)", scale: 0.985 }}
      transition={{ duration: 0.32, ease: "easeOut" }}
    >
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[24px] font-bold tracking-tight text-[var(--text-primary)]">{title}</h1>
          {subtitle && (
            <p className="mt-1 text-[13px] text-[var(--text-muted)]">{subtitle}</p>
          )}
        </div>
        {badges && badges.length > 0 && (
          <div className="flex items-center gap-2">
            {badges.map((b) => (
              <span
                key={b.label}
                className="flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-medium"
                style={{ borderColor: "var(--border-strong)", background: "rgba(255,255,255,0.02)" }}
              >
                {b.tone && <StatusDot status={b.tone} size={5} />}
                <span className="text-[var(--text-secondary)]">{b.label}</span>
              </span>
            ))}
          </div>
        )}
      </div>
      {children}
    </motion.div>
  )
}
