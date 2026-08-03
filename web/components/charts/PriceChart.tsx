"use client"

import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { ChartPoint } from "@/lib/types"

type Props = {
  points: ChartPoint[]
  height?: number
  showSMA?: boolean
  showEMA?: boolean
  showBands?: boolean
  color?: string
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

function ChartTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload?.length) return null
  return (
    <div
      className="rounded-xl border p-3 backdrop-blur-xl"
      style={{
        background: "rgba(17,17,24,0.92)",
        borderColor: "var(--border-strong)",
        boxShadow: "0 12px 40px rgba(0,0,0,0.5)",
      }}
    >
      <div className="mb-1.5 font-mono text-[11px] text-[var(--text-muted)]">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-6">
          <span className="text-[11px] uppercase tracking-wide" style={{ color: p.color }}>
            {p.name}
          </span>
          <span className="font-mono text-[12px] text-[var(--text-primary)]">
            {typeof p.value === "number" ? p.value.toFixed(2) : "—"}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function PriceChart({
  points,
  height = 360,
  showSMA = true,
  showEMA = false,
  showBands = true,
  color = "#38bdf8",
}: Props) {
  const up = points.length > 1 && points[points.length - 1].close >= points[0].close
  const lineColor = up ? "var(--up)" : "var(--down)"
  const useColor = showSMA ? lineColor : color
  const data = points.map((p) => ({ ...p }))

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 4, bottom: 0, left: 4 }}>
          <defs>
            <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={useColor} stopOpacity={0.18} />
              <stop offset="100%" stopColor={useColor} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="smaFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2f6fed" stopOpacity={0.12} />
              <stop offset="100%" stopColor="#2f6fed" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            stroke="rgba(255,255,255,0.05)"
            strokeDasharray="2 6"
            vertical={false}
          />
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
            width={52}
            tickFormatter={(v: number) => v.toFixed(0)}
          />
          <Tooltip content={<ChartTooltip />} cursor={{ stroke: "rgba(255,255,255,0.15)" }} />
          {showBands && (
            <>
              <Area
                type="monotone"
                dataKey="bb_upper"
                name="BB Upper"
                stroke="rgba(255,255,255,0.16)"
                strokeWidth={1}
                strokeDasharray="3 4"
                fill="none"
                isAnimationActive={false}
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="bb_lower"
                name="BB Lower"
                stroke="rgba(255,255,255,0.16)"
                strokeWidth={1}
                strokeDasharray="3 4"
                fill="none"
                isAnimationActive={false}
                dot={false}
              />
            </>
          )}
          {showSMA && (
            <>
              <Line
                type="monotone"
                dataKey="sma_20"
                name="SMA 20"
                stroke="#2f6fed"
                strokeWidth={1.2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="sma_50"
                name="SMA 50"
                stroke="#d9a441"
                strokeWidth={1.2}
                dot={false}
                isAnimationActive={false}
              />
            </>
          )}
          {showEMA && (
            <>
              <Line
                type="monotone"
                dataKey="ema_12"
                name="EMA 12"
                stroke="#a78bfa"
                strokeWidth={1.2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="ema_26"
                name="EMA 26"
                stroke="#f472b6"
                strokeWidth={1.2}
                dot={false}
                isAnimationActive={false}
              />
            </>
          )}
          <Area
            type="monotone"
            dataKey="close"
            name="Close"
            stroke={useColor}
            strokeWidth={2}
            fill="url(#priceFill)"
            dot={false}
            animationDuration={900}
            className="chart-glow"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
