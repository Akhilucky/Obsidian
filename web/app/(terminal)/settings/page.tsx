"use client"

import { useEffect, useState } from "react"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import { api } from "@/lib/api"
import type { Health, SettingsInfo } from "@/lib/types"

export default function SettingsPage() {
  const [health, setHealth] = useState<Health | null>(null)
  const [info, setInfo] = useState<SettingsInfo | null>(null)
  const [cleared, setCleared] = useState(false)

  const refresh = () => {
    const [h, s] = [api.health(), api.settings()]
    h.then(setHealth).catch(() => {})
    s.then(setInfo).catch(() => {})
  }

  useEffect(() => {
    refresh()
  }, [])

  const clear = async () => {
    await api.clearCache()
    setCleared(true)
    setTimeout(() => setCleared(false), 2500)
    refresh()
  }

  const rows = [
    { label: "Cache Location", value: info?.cache_location ?? "—" },
    { label: "Cached Files", value: String(info?.cached_files ?? 0) },
    { label: "Frontend", value: "React Terminal" },
    { label: "Design System", value: info?.design_system ?? "Obsidian Institutional" },
    { label: "Chart Engine", value: info?.chart_engine ?? "Recharts" },
    { label: "Backend", value: info?.framework ?? "Flask API" },
    { label: "Agent Engine", value: info?.agent_engine ?? "Obsidian Orchestrator" },
    { label: "Compute Kernels", value: "C++ (fast kernels) + Java (optimizer)" },
    { label: "API Status", value: health?.status ?? "…" },
    { label: "Cached Entries", value: String(health?.cached_entries ?? 0) },
  ]

  return (
    <Page
      title="Settings"
      subtitle="Configuration & system information"
      badges={[{ label: health?.market_open ? "Market Open" : "Market Closed", tone: health?.market_open ? "up" : "neutral" }]}
    >
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card title="Data Management">
          <p className="mb-4 text-[12px] leading-relaxed text-[var(--text-muted)]">
            Clear the API cache to force fresh market data on the next request.
            Parquet cache files are also removed from disk.
          </p>
          <button
            onClick={clear}
            className="flex items-center gap-2 rounded-xl border px-4 py-2.5 text-[13px] font-medium transition-all duration-150 hover:brightness-110 active:scale-[0.98]"
            style={{
              borderColor: "rgba(228,87,61,0.4)",
              background: "rgba(228,87,61,0.08)",
              color: "var(--down)",
            }}
          >
            {cleared ? "✓ Cache cleared" : "Clear Cache & Refresh"}
          </button>
        </Card>

        <Card title="System Information">
          <div className="space-y-0">
            {rows.map((r) => (
              <div
                key={r.label}
                className="flex items-center justify-between border-b py-2.5 last:border-b-0"
                style={{ borderColor: "rgba(255,255,255,0.04)" }}
              >
                <span className="text-[12px] text-[var(--text-muted)]">{r.label}</span>
                <span className="font-mono text-[12px] text-[var(--text-primary)]">{r.value}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Runtime" className="mt-5">
        <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
          <RuntimeItem label="Engine" value={health?.engine ?? "…"} />
          <RuntimeItem label="Backend" value={health?.backend ?? "…"} />
          <RuntimeItem label="Server Time" value={health?.time ?? "…"} />
          <RuntimeItem label="Status" value={health?.status === "ok" ? "● Operational" : "…"} />
        </div>
      </Card>
    </Page>
  )
}

function RuntimeItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">{label}</div>
      <div className="mt-1 font-mono text-[14px] text-[var(--text-primary)]">{value}</div>
    </div>
  )
}
