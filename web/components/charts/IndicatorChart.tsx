"use client"

import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { ChartPoint } from "@/lib/types"

type Props = {
  points: ChartPoint[]
  kind: "rsi" | "macd"
  height?: number
}

type TooltipEntry = {
  dataKey?: string
  name?: string
  value?: number
  color?: string
}

type TooltipProps = {
  active?: boolean
  payload?: TooltipEntry[]
  label?: string
}

function MiniTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload?.length) return null
  return (
    <div
      className="rounded-lg border px-2.5 py-1.5 backdrop-blur-xl"
      style={{ background: "rgba(17,17,24,0.92)", borderColor: "var(--border-strong)" }}
    >
      <div className="font-mono text-[10px] text-[var(--text-muted)]">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="font-mono text-[11px]" style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(2) : "—"}
        </div>
      ))}
    </div>
  )
}

export default function IndicatorChart({ points, kind, height = 140 }: Props) {
  const data = points.map((p) => ({ ...p }))
  const grid = (
    <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="2 6" vertical={false} />
  )
  const axes = (
    <>
      <XAxis
        dataKey="date"
        tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 9 }}
        axisLine={false}
        tickLine={false}
        minTickGap={60}
      />
      <YAxis
        width={34}
        tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 9 }}
        axisLine={false}
        tickLine={false}
      />
    </>
  )

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        {kind === "rsi" ? (
          <LineChart data={data} margin={{ top: 6, right: 4, bottom: 0, left: 0 }}>
            {grid}
            {axes}
            <Tooltip content={<MiniTooltip />} />
            <Line
              type="monotone"
              dataKey="rsi"
              name="RSI"
              stroke="#a78bfa"
              strokeWidth={1.5}
              dot={false}
              animationDuration={700}
            />
          </LineChart>
        ) : (
          <BarChart data={data} margin={{ top: 6, right: 4, bottom: 0, left: 0 }}>
            {grid}
            {axes}
            <Tooltip content={<MiniTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
            <Bar
              dataKey="macd_hist"
              name="Hist"
              fill="#38bdf8"
              opacity={0.55}
              animationDuration={700}
              radius={[2, 2, 0, 0]}
            />
            <Line
              type="monotone"
              dataKey="macd"
              name="MACD"
              stroke="#38bdf8"
              strokeWidth={1.4}
              dot={false}
              animationDuration={700}
            />
            <Line
              type="monotone"
              dataKey="macd_signal"
              name="Signal"
              stroke="#d9a441"
              strokeWidth={1.2}
              dot={false}
              animationDuration={700}
            />
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  )
}
