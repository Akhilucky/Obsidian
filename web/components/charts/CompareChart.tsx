"use client"

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { CompareResult } from "@/lib/types"

const COLORS = ["#38bdf8", "#34c88a", "#d9a441", "#a78bfa", "#f472b6", "#22d3ee", "#e4573d", "#818cf8"]

type Props = {
  data: CompareResult
  height?: number
}

type TooltipEntry = {
  dataKey?: string
  value?: number
  color?: string
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: TooltipEntry[]; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div
      className="rounded-xl border p-3 backdrop-blur-xl"
      style={{ background: "rgba(17,17,24,0.92)", borderColor: "var(--border-strong)", boxShadow: "0 12px 40px rgba(0,0,0,0.5)" }}
    >
      <div className="mb-1.5 font-mono text-[11px] text-[var(--text-muted)]">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-6">
          <span className="text-[11px] uppercase tracking-wide" style={{ color: p.color }}>
            {p.dataKey}
          </span>
          <span className="font-mono text-[12px] text-[var(--text-primary)]">
            {typeof p.value === "number" ? p.value.toFixed(2) : "—"}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function CompareChart({ data, height = 300 }: Props) {
  const tickers = data.tickers ?? []
  const maxLen = Math.max(...tickers.map((t) => data.series[t]?.length ?? 0), 0)
  const rows: Record<string, number | string>[] = []
  for (let i = 0; i < maxLen; i++) {
    const row: Record<string, number | string> = { date: "" }
    for (const t of tickers) {
      const p = data.series[t]?.[i]
      if (p) {
        row["date"] = p.date
        row[t] = p.value
      }
    }
    if (row["date"]) rows.push(row)
  }

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 8, right: 4, bottom: 0, left: 4 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="2 6" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            minTickGap={48}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            width={48}
            tickFormatter={(v: number) => v.toFixed(0)}
          />
          <Tooltip content={<ChartTooltip />} cursor={{ stroke: "rgba(255,255,255,0.15)" }} />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "var(--text-muted)" }}
            iconType="plainline"
          />
          {tickers.map((t, i) => (
            <Line
              key={t}
              type="monotone"
              dataKey={t}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
